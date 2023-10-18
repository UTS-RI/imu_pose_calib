#include <iostream>
#include <string>


//#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/normal_prior.h>

#include "utils.h"



const double kExpNormTolerance = 1e-14;

inline int getIdJustBefore(double t, const std::vector<double>& timestamps)
{
    int id = -1;
    for(int i = 0; i < timestamps.size()-1; i++)
    {
        if((t >= timestamps[i]) && (t < timestamps[i+1]))
        {
            return i;
        }
    }
    return id;
}

inline Eigen::Matrix3d jacobianRighthandSO3(const Eigen::Vector3d& vec)
{
    Eigen::Matrix3d output = Eigen::Matrix3d::Identity();
    double vec_norm = vec.norm();

    if(vec_norm > kExpNormTolerance)
    {

        Eigen::Matrix3d skew_mat;
        skew_mat << 0.0, -vec(2), vec(1),
                    vec(2), 0.0, -vec(0),
                    -vec(1), vec(0), 0.0;
        
        output += ( (vec_norm - sin(vec_norm)) / (vec_norm*vec_norm*vec_norm) )*skew_mat*skew_mat  - ( (1.0 - cos(vec_norm))/(vec_norm*vec_norm) )*skew_mat;
    }
    return output;
}
// Lefthand Jacobian of SO3 Exp mapping
inline Eigen::Matrix3d jacobianLefthandSO3(const Eigen::Vector3d& rot_vec)
{
    return jacobianRighthandSO3(-rot_vec);
}

inline Eigen::MatrixXd seKernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double l2, const double sf2)
{
    Eigen::MatrixXd D2(x1.size(), x2.size());
    for(int i = 0; i < x2.size(); i++)
    {
        D2.col(i) = (x1.array() - x2(i)).square();
    }
    return ((D2 * (-0.5/l2)).array().exp() * sf2).matrix();
}

inline Eigen::MatrixXd seKerneldt(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double l2, const double sf2)
{
    Eigen::MatrixXd D2(x1.size(), x2.size());
    Eigen::MatrixXd D(x1.size(), x2.size());
    for(int i = 0; i < x2.size(); i++)
    {
        D.col(i) = (x1.array() - x2(i));
        D2.col(i) = (x1.array() - x2(i)).square();
    }
    return -((D2 * (-0.5/l2)).array().exp() * sf2 * (D.array())).matrix()/l2;
}

//( k*4*(x-z)*(x - z)/(4*l2^2) - k/l2 )
inline Eigen::MatrixXd seKerneldt2(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double l2, const double sf2)
{
    Eigen::MatrixXd D2(x1.size(), x2.size());
    Eigen::MatrixXd D(x1.size(), x2.size());
    for(int i = 0; i < x2.size(); i++)
    {
        D.col(i) = (x1.array() - x2(i));
        D2.col(i) = (x1.array() - x2(i)).square();
    }
    Eigen::MatrixXd K = ((D2 * (-0.5/l2)).array().exp() * sf2).matrix();
    return (K.array()*4*(D2.array())/ (4*l2*l2) - K.array()/l2).matrix();
}



//( k*(x-z)*( -(x-z)*(x-z) + 3*l2  )/(l2*l2*l2) )
inline Eigen::MatrixXd seKerneldt3(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const double l2, const double sf2)
{
    Eigen::MatrixXd D2(x1.size(), x2.size());
    Eigen::MatrixXd D(x1.size(), x2.size());
    for(int i = 0; i < x2.size(); i++)
    {
        D.col(i) = (x1.array() - x2(i));
        D2.col(i) = (x1.array() - x2(i)).square();
    }
    Eigen::MatrixXd K = ((D2 * (-0.5/l2)).array().exp() * sf2).matrix();
    return (K.array()*(D.array())*(-D2.array() + 3*l2)/ (l2*l2*l2)).matrix();
}


inline Eigen::MatrixXd cholesky(const Eigen::MatrixXd& K)
{
    Eigen::MatrixXd L(K.rows(), K.cols());
    Eigen::LLT<Eigen::MatrixXd> lltOfA(K);
    L = lltOfA.matrixL();

    return L;
}

inline Eigen::VectorXd solveKinvY(const Eigen::MatrixXd& K, const Eigen::VectorXd& Y)
{
    Eigen::MatrixXd L = cholesky(K);
    Eigen::VectorXd alpha;
    alpha = L.triangularView<Eigen::Lower>().solve(Y);
    L.triangularView<Eigen::Lower>().transpose().solveInPlace(alpha);

    return alpha;
}


class GPR3D
{
    private:
        Eigen::MatrixXd alphas_;
        Eigen::VectorXd timestamps_;
        double l2_;
        Eigen::Vector3d mean_;
        Eigen::Vector3d sf2_;
        Eigen::Vector3d sz2_;


        void computeHyper(const Eigen::MatrixXd& val)
        {
            int smooth_size=10;

            for(int c = 0; c < 3; ++c)
            {
                double mean = val.col(c).mean();

                mean_(c) = mean;
                sf2_(c) = 1.0;
                sz2_(c) = 0.01;

                //// Experimental hyperparameter estimation (does not work well)
                //double sf2 = 0.0;
                //double sz2 = 0.0;
                //for(int i = 0; i < val.rows(); ++i)
                //{
                //    sf2 += (val(i,c) - mean)*(val(i,c) - mean);
                //}
                //sf2 /= val.rows();

                //Eigen::VectorXd vec_smoothed(val.rows()-2*smooth_size);
                //for(int i = smooth_size; i < val.rows()-smooth_size; ++i)
                //{
                //    vec_smoothed(i-smooth_size) = val.col(c).segment(i-smooth_size, 2*smooth_size+1).mean();
                //}

                //sz2 = vec_smoothed.array().square().mean();

                //sf2_(c) = sf2;
                //sz2_(c) = sz2;

            }
        }
    public:
        GPR3D(const Eigen::VectorXd& t, const Eigen::MatrixXd& y, const double lengthscale_factor)
        {
            timestamps_ = t;
            // Get the median pose period
            std::vector<double> pose_period;
            for(int i = 0; i < t.size()-1; ++i)
            {
                pose_period.push_back(t(i+1) - t(i));
            }
            std::sort(pose_period.begin(), pose_period.end());
            double median_pose_period = pose_period[pose_period.size()/2];
            l2_ = lengthscale_factor*lengthscale_factor*median_pose_period*median_pose_period;
            computeHyper(y);

            alphas_ = Eigen::MatrixXd::Zero(t.size(), 3);
            Eigen::MatrixXd K_base = seKernel(t, t, l2_, 1.0);
            for (int c = 0; c < 3; ++c)
            {
                alphas_.col(c) = solveKinvY(sf2_(c)*K_base + sz2_(c)*Eigen::MatrixXd::Identity(t.size(), t.size()), (y.col(c).array() - mean_(c)).matrix());
            }
        }

        Eigen::Vector3d query(const double t, const int derivative_order=0) const
        {
            Eigen::VectorXd t_eig(1);
            t_eig(0) = t;
            Eigen::MatrixXd ks;
            if (derivative_order == 0)
                ks = seKernel(t_eig, timestamps_, l2_, 1.0);
            else if (derivative_order == 1)
                ks = seKerneldt(t_eig, timestamps_, l2_, 1.0);
            else if (derivative_order == 2)
                ks = seKerneldt2(t_eig, timestamps_, l2_, 1.0);
            else
                std::cout << "Error GPR3D: derivative order not implemented" << std::endl;
            Eigen::Vector3d output;
            output(0) = (sf2_(0)*ks*alphas_.col(0))(0,0);
            output(1) = (sf2_(1)*ks*alphas_.col(1))(0,0);
            output(2) = (sf2_(2)*ks*alphas_.col(2))(0,0);
            if (derivative_order == 0)
                output += mean_;
            return output;
        }

};

class AccCostFunction : public ceres::SizedCostFunction<3, 3, 3, 3, 3, 1>
{
    private:
        Eigen::Vector3d acc_meas_;
        double t_;
        Eigen::Vector3d g_;
        double weight_;

        std::shared_ptr<GPR3D> pos_gp_;
        std::shared_ptr<GPR3D> rot_gp_;
        double l2_, sf2_;


    public:
        AccCostFunction(const Eigen::Vector3d& acc_meas, double t, double weight, const Eigen::Vector3d& g, const std::shared_ptr<GPR3D> pos_gp, const std::shared_ptr<GPR3D> rot_gp)
            : acc_meas_(acc_meas)
            , t_(t)
            , g_(g)
            , weight_(weight)
            , pos_gp_(pos_gp)
            , rot_gp_(rot_gp)
        {
        }

        bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        {
            // Unpack parameters
            Eigen::Map<const Eigen::Vector3d> r_0(parameters[0]);
            Eigen::Map<const Eigen::Vector3d> r_c(parameters[1]);
            Eigen::Map<const Eigen::Vector3d> p_c(parameters[2]);
            Eigen::Map<const Eigen::Vector3d> acc_bias(parameters[3]);
            const double dt = parameters[4][0];

            Eigen::Vector3d unbiased_acc = acc_meas_ - acc_bias;
            Eigen::Vector3d rot_acc;
            ceres::AngleAxisRotatePoint(r_c.data(), unbiased_acc.data(), rot_acc.data());

            const double time_corrected = t_ + dt;

            // Get the different parts of the pose state
            Eigen::Vector3d ang_vel = rot_gp_->query(time_corrected, 1);
            Eigen::Vector3d lin_acc = pos_gp_->query(time_corrected, 2);
            Eigen::Vector3d ang_acc = rot_gp_->query(time_corrected, 2);
            Eigen::Vector3d rot_vec = rot_gp_->query(time_corrected, 0);

            Eigen::Quaterniond q(Eigen::AngleAxisd(rot_vec.norm(), rot_vec.normalized()));



            Eigen::Quaterniond q_0(Eigen::AngleAxisd(r_0.norm(), r_0.normalized()));
            Eigen::Matrix3d R_0 = q_0.toRotationMatrix();
            Eigen::Vector3d R0T_g = R_0.transpose()*g_;
            Eigen::Matrix3d R = q.toRotationMatrix();
            Eigen::Vector3d RT_R0T_g = R.transpose()*R0T_g;

            lin_acc = R.transpose()*lin_acc;
            ang_acc = R.transpose()*ang_acc;

            Eigen::Map<Eigen::Vector3d> res(residuals);
            res = weight_ * (rot_acc - (lin_acc) - (ang_acc.cross(p_c)) - R.transpose()*(ang_vel.cross(ang_vel.cross(p_c))) + RT_R0T_g);

            if(jacobians != NULL)
            {
                if(jacobians[0] != NULL)
                {
                    Eigen::Matrix3d R0T_g_skew;
                    R0T_g_skew << 0, -R0T_g(2), R0T_g(1),
                                   R0T_g(2), 0, -R0T_g(0),
                                   -R0T_g(1), R0T_g(0), 0;
                    Eigen::Matrix3d d_R0T_g_d_r_0 =  R0T_g_skew*jacobianLefthandSO3(-r_0);

                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_r_0(jacobians[0]);
                    jacobian_r_0 = weight_ * (R.transpose())* d_R0T_g_d_r_0;
                }
                if(jacobians[1] != NULL)
                {
                    Eigen::Matrix3d rot_acc_skew;
                    rot_acc_skew << 0, -rot_acc(2), rot_acc(1),
                                    rot_acc(2), 0, -rot_acc(0),
                                    -rot_acc(1), rot_acc(0), 0;

                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_r_c(jacobians[1]);
                    jacobian_r_c = -weight_*rot_acc_skew*jacobianLefthandSO3(r_c);

                }
                if(jacobians[2] != NULL)
                {
                    Eigen::Matrix3d ang_acc_skew;
                    ang_acc_skew << 0, -ang_acc(2), ang_acc(1),
                                    ang_acc(2), 0, -ang_acc(0),
                                    -ang_acc(1), ang_acc(0), 0;
                    Eigen::Matrix3d ang_vel_skew;
                    ang_vel_skew << 0, -ang_vel(2), ang_vel(1),
                                    ang_vel(2), 0, -ang_vel(0),
                                    -ang_vel(1), ang_vel(0), 0;

                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_p_c(jacobians[2]);
                    jacobian_p_c = weight_*(-ang_acc_skew - R.transpose()*ang_vel_skew*ang_vel_skew);
                }
                if(jacobians[3] != NULL)
                {
                    ceres::Matrix R_c(3, 3);
                    ceres::AngleAxisToRotationMatrix(r_c.data(), R_c.data());
                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_bias(jacobians[3]);
                    jacobian_bias = -weight_*R_c.transpose();
                }
                if(jacobians[4] != NULL)
                {
                    double quatum = 0.000001;
                    const double dt_shift = dt+quatum;
                    const double* temp_param[5] = {&parameters[0][0], &parameters[1][0], &parameters[2][0], &parameters[3][0], &dt_shift};
                    Eigen::Vector3d temp_res;
                    Evaluate(temp_param, temp_res.data(), NULL);

                    Eigen::Map<Eigen::Matrix<double, 3, 1>> jacobian_dt(jacobians[4]);
                    jacobian_dt = (temp_res - res)/quatum;

                }

            }
            
            return true;
        }

};



class RotCostFunction : public ceres::SizedCostFunction<3, 3, 3, 1>
{
    private:
        double gyr_x_, gyr_y_, gyr_z_, gyr_t_;
        double weight_;
        std::shared_ptr<GPR3D> rot_gp_;

    public:

        RotCostFunction(const Eigen::Vector3d& gyr, const double time, const double weight, const std::shared_ptr<GPR3D> rot_gp)
                : gyr_x_(gyr(0))
                , gyr_y_(gyr(1))
                , gyr_z_(gyr(2))
                , gyr_t_(time  )
                , rot_gp_(rot_gp)
                , weight_(weight)
        {

        }

        virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        {
            // Unstack the parameters
            const double r_c[3] = {parameters[0][0], parameters[0][1], parameters[0][2]};
            const double bias[3] = {parameters[1][0], parameters[1][1], parameters[1][2]};
            const double dt = parameters[2][0];


            // Remove the biases from the gyr data
            double gyr[3];
            gyr[0] = gyr_x_ - bias[0];
            gyr[1] = gyr_y_ - bias[1];
            gyr[2] = gyr_z_ - bias[2];

            // Rotate the biases given the state
            double rot_gyr[3];
            ceres::AngleAxisRotatePoint(r_c, gyr, rot_gyr);

            // Compute the corrected time stamp of the IMU
            const double corrected_time = gyr_t_ + dt;

            // Querry the pose_ang_vel
            Eigen::Vector3d rot_vec = rot_gp_->query(corrected_time, 0);
            Eigen::Vector3d d_rot_vec = rot_gp_->query(corrected_time, 1);

            Eigen::Vector3d pose_ang_vel = jacobianRighthandSO3(rot_vec)*d_rot_vec;


            // Compute the residual as the difference between the rotated gyroscope measurements and the queried pose_ang_vel
            residuals[0] = weight_*(rot_gyr[0] - pose_ang_vel(0));
            residuals[1] = weight_*(rot_gyr[1] - pose_ang_vel(1));
            residuals[2] = weight_*(rot_gyr[2] - pose_ang_vel(2));

            if (jacobians != NULL)
            {
                if (jacobians[0] != NULL)
                {
                    Eigen::Matrix3d rot_gyr_skew;
                    rot_gyr_skew << 0, -rot_gyr[2], rot_gyr[1],
                            rot_gyr[2], 0, -rot_gyr[0],
                            -rot_gyr[1], rot_gyr[0], 0;

                    Eigen::Map<const Eigen::Vector3d> r_c_vec(r_c);
                    Eigen::Matrix3d d_rot_gyr_d_r_c = - rot_gyr_skew*jacobianLefthandSO3(r_c_vec);

                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_r_c(jacobians[0]);
                    jacobian_r_c = weight_*d_rot_gyr_d_r_c;

                }
                if (jacobians[1] != NULL)
                {
                    ceres::Matrix R_c(3, 3);
                    ceres::AngleAxisToRotationMatrix(r_c, R_c.data());
                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_bias(jacobians[1]);
                    jacobian_bias = -weight_*R_c.transpose();
                }
                if (jacobians[2] != NULL)
                {
                    double quatum = 0.000001;
                    const double dt_shift = dt+quatum;
                    const double* temp_param[3] = {&parameters[0][0], &parameters[1][0], &dt_shift};
                    double temp_res[3];
                    Evaluate(temp_param, temp_res, NULL);

                    Eigen::Map<Eigen::Matrix<double, 3, 1>> jacobian_dt(jacobians[2]);
                    jacobian_dt[0] = (temp_res[0] - residuals[0])/quatum;
                    jacobian_dt[1] = (temp_res[1] - residuals[1])/quatum;
                    jacobian_dt[2] = (temp_res[2] - residuals[2])/quatum;

                }
            }
            return true;
        }
                    
};


class BiasCostFunction : public ceres::SizedCostFunction<3, 3, 3>
{
    private:
        double weight_;

    public:
        BiasCostFunction(const double weight)
                : weight_(weight)
        {}
    
        bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
        {
            // Compute the residual as the difference between the two states
            residuals[0] = weight_*(parameters[0][0] - parameters[1][0]);
            residuals[1] = weight_*(parameters[0][1] - parameters[1][1]);
            residuals[2] = weight_*(parameters[0][2] - parameters[1][2]);

            // Compute the jacobians
            if (jacobians != NULL)
            {
                if (jacobians[0] != NULL)
                {
                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_bias_0(jacobians[0]);
                    jacobian_bias_0 = weight_*Eigen::Matrix3d::Identity();
                }
                if (jacobians[1] != NULL)
                {
                    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_bias_1(jacobians[1]);
                    jacobian_bias_1 = -weight_*Eigen::Matrix3d::Identity();
                }
            }
            return true;
        }
};


int main(int argc, char* argv[])
{

    std::cout << "Reading data (c++)" << std::endl;
    Eigen::MatrixXd acc_data = readCSV("../temp/acc_data.csv");
    Eigen::MatrixXd gyr_data = readCSV("../temp/gyr_data.csv");
    Eigen::MatrixXd acc_bias_weights = readCSV("../temp/acc_bias_weights.csv");
    Eigen::MatrixXd gyr_bias_weights = readCSV("../temp/gyr_bias_weights.csv");
    Eigen::MatrixXd acc_weights = readCSV("../temp/acc_weights.csv");
    Eigen::MatrixXd gyr_weights = readCSV("../temp/gyr_weights.csv");
    Eigen::MatrixXd rot_prior = readCSV("../temp/rot_prior.csv");
    Eigen::MatrixXd pos_prior = readCSV("../temp/pos_prior.csv");
    Eigen::MatrixXd pose_data = readCSV("../temp/pose.csv");
    Eigen::MatrixXd gravity = readCSV("../temp/gravity.csv");
    

    std::shared_ptr<GPR3D> pos_gp(new GPR3D(pose_data.col(0), pose_data.block(0,1,pose_data.rows(), 3), 10.0));
    std::shared_ptr<GPR3D> rot_gp(new GPR3D(pose_data.col(0), pose_data.block(0,4,pose_data.rows(), 3), 10.0));


    int nb_acc = acc_data.rows();
    int nb_gyr = gyr_data.rows();

    std::cout << " Number of acc measurements: " << nb_acc << std::endl;
    std::cout << " Number of gyr measurements: " << nb_gyr << std::endl;


    // Average the first 10 imu readings to initialise the biases (assumption that not moving much at the begining of dataset)
    int temp_num = 10;
    Eigen::Vector3d bias_init = Eigen::Vector3d::Zero();
    for(int i = 0; i < temp_num; ++i)
        bias_init += gyr_data.block<1,3>(i,1).transpose();
    bias_init /= temp_num;

    std::cout << "Initial bias: " << bias_init.transpose() << std::endl;


    // Get the median pose period
    std::vector<double> pose_period;
    for(int i = 0; i < pose_data.rows()-1; ++i)
    {
        pose_period.push_back(pose_data(i+1,0) - pose_data(i,0));
    }
    std::sort(pose_period.begin(), pose_period.end());
    double median_pose_period = pose_period[pose_period.size()/2];
    double l2 = 3.0*3.0*median_pose_period*median_pose_period;
    double sf2 = 1.0;
    double sz2 = 0.0001;




    ///////// Optimise for the extrinsic rotation /////////
    std::cout << std::endl << "Optimising for the extrinsic rotation first" << std::endl;
    std::vector<std::array<double,3> > gyr_biases(nb_gyr, {bias_init(0), bias_init(1), bias_init(2)});
    ceres::Problem problem;
    std::array<double, 3> rot_calib;
    rot_calib[0] = rot_prior(0);
    rot_calib[1] = rot_prior(1);
    rot_calib[2] = rot_prior(2);
    double dt = 0;

    // Add the rotation related residuals   
    for(int i = 0; i < gyr_data.rows(); ++i)
    {
        Eigen::Vector3d gyr = gyr_data.block<1,3>(i,1).transpose();
        double time = gyr_data(i,0);

        ceres::CostFunction* rot_cost_function = new RotCostFunction(gyr, time, gyr_weights(i), rot_gp);
        problem.AddResidualBlock(rot_cost_function, NULL, rot_calib.data(), gyr_biases[i].data(), &dt);
    }
    // Add the bias related residuals
    for(int i = 1; i < gyr_data.rows(); ++i)
    {
        ceres::CostFunction* bias_cost_function = new BiasCostFunction(gyr_bias_weights(i-1));
        problem.AddResidualBlock(bias_cost_function, NULL, gyr_biases[i-1].data(), gyr_biases[i].data());
    }

    // First solve with the time shift fixed
    problem.SetParameterBlockConstant(&dt);

    ceres::Solver::Options opts;
    opts.minimizer_progress_to_stdout = false;
    opts.max_num_iterations = 200;
    opts.num_threads = 14;
    ceres::Solver::Summary rot_summary;
    ceres::Solve(opts, &problem, &rot_summary);


    // Set the time shift dt to be free
    problem.SetParameterBlockVariable(&dt);

    ceres::Solve(opts, &problem, &rot_summary);

    

    std::cout << "Rot estimate " << rot_calib[0] << " " << rot_calib[1] << " " << rot_calib[2] << std::endl;
    std::cout << "dt estimate " << dt << std::endl;



    /////////// Optimise for the extrinsic position /////////
    std::cout << std::endl << "Optimising for the full extrinsic pose" << std::endl;
    std::array<double, 3> pos_calib;
    pos_calib[0] = pos_prior(0);
    pos_calib[1] = pos_prior(1);
    pos_calib[2] = pos_prior(2);

    // Compute the first orientation prior
    std::array<double, 3> rot_0;
    // Get the average of the first 10 acc measurements
    Eigen::Vector3d acc_0 = Eigen::Vector3d::Zero();
    for(int i = 0; i < temp_num; ++i)
        acc_0 += acc_data.block<1,3>(i,1);
    // Normalize the acc
    acc_0.normalize();
    Eigen::Vector3d g_unit = -(gravity.normalized());
    Eigen::Quaterniond imu_q_0 = Eigen::Quaterniond::FromTwoVectors(acc_0, g_unit);
    Eigen::Matrix3d imu_R_0 = imu_q_0.toRotationMatrix();

    //// Query the rotation at timestamp of the first acc measurement
    std::vector<double> temp_time(pose_data.rows());
    for(int i = 0; i < pose_data.rows(); ++i)
        temp_time[i] = pose_data(i,0);
    int id = getIdJustBefore(acc_data(0,0), temp_time);
    double alpha = (acc_data(0,0) - pose_data(id,0))/(pose_data(id+1,0) - pose_data(id,0));
    // Interpolate the rotation vector at the timestamp of the first acc measurement
    Eigen::Vector3d pose_r_0 = pose_data.block<1,3>(id,4).transpose()*(1-alpha) + pose_data.block<1,3>(id+1,4).transpose()*alpha;
    Eigen::Matrix3d pose_R_0 = Eigen::AngleAxisd(pose_r_0.norm(), pose_r_0.normalized()).toRotationMatrix();

    Eigen::Vector3d r_c_vec(rot_calib[0], rot_calib[1], rot_calib[2]);
    Eigen::Matrix3d R_calib = Eigen::AngleAxisd(r_c_vec.norm(), r_c_vec.normalized()).toRotationMatrix();
    ceres::Matrix R_0 = imu_R_0*(R_calib.transpose())*(pose_R_0.transpose());
    ceres::RotationMatrixToAngleAxis(R_0.data(), rot_0.data());


    std::vector<std::array<double, 3>> pos_biases;
    pos_biases.resize(acc_data.rows());
    for(int i = 0; i < acc_data.rows(); ++i)
    {
        pos_biases[i][0] = 0.0;
        pos_biases[i][1] = 0.0;
        pos_biases[i][2] = 0.0;
    }


    // Add the acceleration residuals with a cauchy loss function
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(0.2);
    for(int i = 0; i < acc_data.rows(); ++i)
    {
        Eigen::Vector3d acc = acc_data.block<1,3>(i,1).transpose();
        double time = acc_data(i,0);

        ceres::CostFunction* pos_cost_function = new AccCostFunction(acc, time, acc_weights(i), gravity, pos_gp, rot_gp);
        problem.AddResidualBlock(pos_cost_function, loss_function, rot_0.data(), rot_calib.data(), pos_calib.data(), pos_biases[i].data(), &dt);
    }

    // Add the the pos bias residuals
    for(int i = 1; i < acc_data.rows(); ++i)
    {
        ceres::CostFunction* pos_bias_cost_function = new BiasCostFunction(acc_bias_weights(i-1));
        problem.AddResidualBlock(pos_bias_cost_function, NULL, pos_biases[i-1].data(), pos_biases[i].data());
    }


    // Set rot calib to be constant
    problem.SetParameterBlockConstant(rot_calib.data());
    // Set dt to be constant
    problem.SetParameterBlockConstant(&dt);
    // Set the gyr biases to be constant
    for(int i = 0; i < gyr_biases.size(); ++i)
        problem.SetParameterBlockConstant(gyr_biases[i].data());


    ceres::Solve(opts, &problem, &rot_summary);


    // Set rot calib to be free
    problem.SetParameterBlockVariable(rot_calib.data());
    // Set dt to be free
    problem.SetParameterBlockVariable(&dt);
    // Set the gyr biases to be constant
    for(int i = 0; i < gyr_biases.size(); ++i)
        problem.SetParameterBlockVariable(gyr_biases[i].data());


    opts.function_tolerance = 1e-10;
    //opts.check_gradients = true;
    ceres::Solve(opts, &problem, &rot_summary);


    // Print the results
    std::cout << "Rotation calibration: " << rot_calib[0] << " " << rot_calib[1] << " " << rot_calib[2] << std::endl;
    std::cout << "Position calibration: " << pos_calib[0] << " " << pos_calib[1] << " " << pos_calib[2] << std::endl;
    std::cout << "dt: " << dt << std::endl;
    std::cout << "Initial rotation (gravity alignment): " << rot_0[0] << " " << rot_0[1] << " " << rot_0[2] << std::endl;

















    //////// Write the results to files ////////

    // Put the biases in a matrix
    Eigen::MatrixXd gyr_biases_mat(gyr_biases.size(), 3);
    for(int i = 0; i < gyr_biases.size(); ++i)
    {
        gyr_biases_mat(i,0) = gyr_biases[i][0];
        gyr_biases_mat(i,1) = gyr_biases[i][1];
        gyr_biases_mat(i,2) = gyr_biases[i][2];
    }
    writeCSV("../temp/gyr_biases.csv", gyr_biases_mat);


    // Put the acc biases in a matrix
    Eigen::MatrixXd acc_biases_mat(pos_biases.size(), 3);
    for(int i = 0; i < pos_biases.size(); ++i)
    {
        acc_biases_mat(i,0) = pos_biases[i][0];
        acc_biases_mat(i,1) = pos_biases[i][1];
        acc_biases_mat(i,2) = pos_biases[i][2];
    }
    writeCSV("../temp/acc_biases.csv", acc_biases_mat);

    
    Eigen::Vector3d rot_calib_vec(rot_calib[0], rot_calib[1], rot_calib[2]);
    writeCSV("../temp/rot_calib.csv", rot_calib_vec);


    Eigen::MatrixXd dt_mat(1,1);
    dt_mat(0,0) = dt;
    writeCSV("../temp/dt.csv", dt_mat);







    Eigen::Vector3d rot_0_vec(rot_0[0], rot_0[1], rot_0[2]);
    writeCSV("../temp/rot_0.csv", rot_0_vec);




    Eigen::Vector3d pos_calib_vec(pos_calib[0], pos_calib[1], pos_calib[2]);
    writeCSV("../temp/pos_calib.csv", pos_calib_vec);



    return 0;
}
