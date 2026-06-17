#include "estimation_tf.h"

#include "ceres_qt.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "estimation_tf_node");
    ros::NodeHandle n;
    // std::ofstream outf;
    std::string s, pkg_path;
    n.getParam("pkg_path", pkg_path);
    n.param<std::string>("pkg_path", pkg_path, "/home/lin/ros_code/calibration_ws/src/lidar2cam_calibration");
    std::string data_path = pkg_path + "/data/data.txt";

    std::string save_path = pkg_path + "/data/result.txt";
    
    // 打开保存的文件,往文件中写入
    std::fstream fs(save_path.c_str(), std::ios::out);

    // 从ROS参数服务器读取相机内参（与params.yaml一致）
    float fx, fy, cx, cy, k1, k2, k3, p1, p2;
    n.param<float>("fx", fx, 617.201);
    n.param<float>("fy", fy, 617.362);
    n.param<float>("cx", cx, 324.637);
    n.param<float>("cy", cy, 242.462);
    n.param<float>("k1", k1, 0.0);
    n.param<float>("k2", k2, 0.0);
    n.param<float>("k3", k3, 0.0);
    n.param<float>("p1", p1, 0.0);
    n.param<float>("p2", p2, 0.0);

    cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << fx, 0., cx,
                                                       0, fy, cy,
                                                       0, 0., 1);
    cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << k1, k2, p1, p2, k3);

    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    std::cout << "\n--- Camera Intrinsics (from params) ---" << std::endl;
    std::cout << "fx=" << fx << " fy=" << fy << " cx=" << cx << " cy=" << cy << std::endl;
    std::cout << "k1=" << k1 << " k2=" << k2 << " p1=" << p1 << " p2=" << p2 << " k3=" << k3 << std::endl;

    // 创建一个点集,用来保存坐标点集合
    std::shared_ptr<PointSet> point_set = std::make_shared<PointSet>();
    
    // 读取data.txt的点坐标
    bool is_read = readPoint(data_path.c_str(), point_set);

    if(!is_read)
    {
        // std::cout << point_set->p_cam_vector_.size() << std::endl;
        std::cout << "\n--Read point is failure..." << std::endl;
        return -1;
    }

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();

    // SVD求解
    estimation_3d3d_SVD(point_set->p_cam_vector_, point_set->p_lidar_vector_, R, t);
    
    Eigen::Matrix3d R_init = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_init = Eigen::Vector3d::Zero();  

     // 3d3d的ceres优化求解
    optimation_3d3d(point_set->p_cam_vector_, point_set->p_lidar_vector_, R_init, t_init);

    Eigen::Vector3d euler;
    rotationMatrix2eulerAngles(R, euler);
    std::cout << "\nSVD 3d3d: \n";
    std::cout << "eulerAngles:\n" << euler.transpose() << std::endl;
    std::cout << "translation:\n" << t.transpose() << std::endl;
    std::cout << "--------------------" << std::endl;
    fs << "# ------SVD-3d3d------\n# eulerAngles:\n" << euler.transpose() << "\n# translation:\n" << t << std::endl;
    
    rotationMatrix2eulerAngles(R_init, euler);
    std::cout << "\nCeres 3d3d: \n";
    std::cout << "eulerAngles:\n" << euler.transpose() << std::endl;
    std::cout << "translation:\n" << t_init.transpose() << std::endl;
    std::cout << "--------------------" << std::endl;
    fs << "\n#------Ceres-3d3d------\n# eulerAngles:\n" << euler.transpose() << "\n# translation:\n" << t_init << std::endl;

    cv::Mat rvec, tvec, rvec_mat; 
    // solvePNP_cv使用从ROS参数读取的内参
    solvePNP_cv(point_set->pts_uv_cv_, point_set->pts_lidar_cv_, rvec, tvec, cameraMatrix, distCoeffs);
    
    cv::Rodrigues(rvec, rvec_mat);
    cv::cv2eigen(rvec_mat, R);
    rotationMatrix2eulerAngles(R, euler);

    std::cout << "\nOpencv-PNP 3d2d: \n";
    std::cout << "eulerAngles:\n" << euler.transpose() << std::endl;
    std::cout << "translation:\n" << tvec << std::endl;
    std::cout << "--------------------" << std::endl;
    fs << "\n#------Opencv-PNP-3d2d------\n# eulerAngles:\n" << euler.transpose() << "\n# translation:\n" << tvec << std::endl;

    Eigen::Quaterniond q1 = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t1 = Eigen::Vector3d::Zero();

    // ceres四元素求解
    lidar2camICP(point_set->p_lidar_vector_, point_set->p_cam_vector_, q1, t1 );

    Eigen::Matrix3d R_cl(q1);
    euler = q1.matrix().eulerAngles(0,1,2);
    euler = euler * 180 / M_PI;

    std::cout << "ceres-q_t-3d3d-RPY(xyz): " << euler.transpose() << std::endl;
    std::cout << t1.transpose() << std::endl;
    fs << "\n#------ceres-q_t-3d3d:------\n# eulerAngles:\n" << euler.transpose() 
        << "\n# translation:\n" << t1 
        << "\n# R_cl_mat :\n" << R_cl.inverse() << std::endl;

    Eigen::Quaterniond q2 = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t2 = Eigen::Vector3d::Zero();

    // Ceres 3d2d使用从ROS参数读取的内参矩阵K
    lidar2cam_2d3d(point_set->p_lidar_vector_, point_set->p_uv_vector_, K, q2, t2);
    
    euler = q2.matrix().eulerAngles(0,1,2);
    euler = euler * 180 / M_PI;

    std::cout << "ceres-q_t-3d2d-RPY(xyz): " << euler.transpose() << std::endl;
    std::cout << t2.transpose() << std::endl;
    fs << "\n#------ceres-q_t-3d2d:------\n# eulerAngles:\n" << euler.transpose() << "\n# translation:\n" << t2 << std::endl;

    fs.close();
    std::cout << "--Save result in " + save_path << "\n" << std::endl;
    return 0;     

}
