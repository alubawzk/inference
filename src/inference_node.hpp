#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <memory>
#include <Eigen/Geometry>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <queue>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float32_multi_array.hpp> 
#include "utils/motion_loader.hpp"
#include "utils/close_chain_mapping.hpp"
#include "utils/thread_pool.hpp"
#include "motor_driver.hpp"
#include "imu_driver.hpp"
#include <std_srvs/srv/trigger.hpp>

class InferenceNode : public rclcpp::Node {
   public:
    InferenceNode() : Node("inference_node") {

        load_config();
        
        thread_pool_ = std::make_unique<ThreadPool>(motor_interface_.size());

        Ort::ThreadingOptions thread_opts;
        if (intra_threads_ > 0) {
            thread_opts.SetGlobalIntraOpNumThreads(intra_threads_);
        }
        env_ = std::make_unique<Ort::Env>(thread_opts, ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeInference");
        if(use_attn_enc_){
            setup_model(normal_ctx_, model_path_, obs_num_ * frame_stack_ + perception_obs_num_);
        } else {
            setup_model(normal_ctx_, model_path_, obs_num_ * frame_stack_);
        }
        if(use_beyondmimic_){
             setup_model(motion_ctx_, motion_model_path_, motion_obs_num_ * motion_frame_stack_);
        }
        active_ctx_ = normal_ctx_.get();

        if(use_beyondmimic_){
            try{
                motion_loader_ = std::make_unique<MotionLoader>(motion_path_);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                exit(1);
            }
        }

        setup_motors();
        setup_imu();

        ankle_decouple_ = std::make_shared<Decouple>();

        obs_ = std::vector<float>(obs_num_, 0.0);
        joint_obs_ = std::vector<float>(joint_num_ * 2, 0.0);
        motion_pos_ = std::vector<float>(joint_num_, 0.0);
        motion_vel_ = std::vector<float>(joint_num_, 0.0);
        cmd_vel_ = std::vector<float>(3, 0.0);
        quat_ = std::vector<float>(4, 0.0);
        ang_vel_ = std::vector<float>(3, 0.0);
        act_ = std::vector<float>(joint_num_, 0.0);
        last_act_ = std::vector<float>(joint_num_, 0.0);
        joint_torques_ = std::vector<float>(joint_num_, 0.0);
        if (use_interrupt_){
            interrupt_action_ = std::vector<float>(10, 0.0);
        }
        if (use_attn_enc_){
            perception_obs_ = std::vector<float>(perception_obs_num_, 0.0);
        }
        reset();

        auto sensor_data_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        auto control_command_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
        joy_subscription_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", sensor_data_qos, std::bind(&InferenceNode::subs_joy_callback, this, std::placeholders::_1));
        cmd_subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", sensor_data_qos, std::bind(&InferenceNode::subs_cmd_callback,this, std::placeholders::_1
        ));
        elevation_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            perception_obs_topic_, sensor_data_qos,
            std::bind(&InferenceNode::subs_elevation_callback, this, std::placeholders::_1));
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_ref_states", sensor_data_qos,
            std::bind(&InferenceNode::subs_joint_state_callback, this, std::placeholders::_1));
        action_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/action", control_command_qos);
        imu_publisher_ =
            this->create_publisher<sensor_msgs::msg::Imu>("/imu", control_command_qos);
        joint_state_publisher_ =
            this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", control_command_qos);
        inference_thread_ = std::thread(&InferenceNode::inference, this);
        timer_pub_ = this->create_wall_timer(std::chrono::milliseconds((int)(dt_ * 1000)),
                                             std::bind(&InferenceNode::apply_action, this));

        reset_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "reset_motors", std::bind(&InferenceNode::reset_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        set_zeros_service_ = this->create_service<std_srvs::srv::Trigger>(
            "set_zeros", std::bind(&InferenceNode::set_zeros_srv, this, std::placeholders::_1, std::placeholders::_2));
        clear_errors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "clear_errors", std::bind(&InferenceNode::clear_errors_srv, this, std::placeholders::_1, std::placeholders::_2));
        read_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "read_motors", std::bind(&InferenceNode::read_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        init_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "init_motors", std::bind(&InferenceNode::init_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        deinit_motors_service_ = this->create_service<std_srvs::srv::Trigger>(
            "deinit_motors", std::bind(&InferenceNode::deinit_motors_srv, this, std::placeholders::_1, std::placeholders::_2));
        start_inference_service_ = this->create_service<std_srvs::srv::Trigger>(
            "start_inference", std::bind(&InferenceNode::start_inference_srv, this, std::placeholders::_1, std::placeholders::_2));
        stop_inference_service_ = this->create_service<std_srvs::srv::Trigger>(
            "stop_inference", std::bind(&InferenceNode::stop_inference_srv, this, std::placeholders::_1, std::placeholders::_2));
    }
    ~InferenceNode() {
        if (inference_thread_.joinable()) {
            inference_thread_.join();
        }
        reset();
        deinit_motors();
        motors_.clear();
        imu_.reset();
    }
    struct ModelContext {
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::MemoryInfo> memory_info;
        std::unique_ptr<Ort::Value> input_tensor;
        std::unique_ptr<Ort::Value> output_tensor;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<const char *> input_names_raw;
        std::vector<const char *> output_names_raw;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
        std::vector<float> input_buffer;
        std::vector<float> output_buffer;
        size_t num_inputs;
        size_t num_outputs;
    };
   private:
    int offline_threshold_ = 10;
    std::atomic<bool> is_init_{false}, is_running_{false}, is_joy_control_{true}, is_interrupt_{false}, is_beyondmimic_{false};
    std::string model_name_, model_path_, motion_name_, motion_path_, motion_model_name_, motion_model_path_, perception_obs_topic_;
    bool use_interrupt_, use_beyondmimic_, use_attn_enc_;
    int obs_num_, motion_obs_num_, perception_obs_num_, frame_stack_, motion_frame_stack_, joint_num_;
    int decimation_;
    std::unique_ptr<Ort::Env> env_;
    int intra_threads_;
    Ort::AllocatorWithDefaultOptions allocator_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr left_leg_publisher_, right_leg_publisher_,
        left_arm_publisher_, right_arm_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_subscription_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr elevation_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr action_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::TimerBase::SharedPtr timer_pub_;
    std::thread inference_thread_;
    float act_alpha_, gyro_alpha_, angle_alpha_;
    float dt_;
    float obs_scales_lin_vel_, obs_scales_ang_vel_, obs_scales_dof_pos_, obs_scales_dof_vel_,
        obs_scales_gravity_b_, clip_observations_;
    float action_scale_, clip_actions_;
    std::vector<long int> usd2urdf_, motor_id_, motor_model_, motor_num_, close_chain_motor_id_, motor_sign_;
    std::vector<double> kp_, kd_, joint_default_angle_, joint_limits_, clip_cmd_;
    std::string motor_type_, imu_type_, motor_interface_type_, imu_interface_type_, imu_interface_;
    std::vector<std::string> motor_interface_;
    int master_id_offset_, imu_id_, baudrate_;
    bool is_first_frame_;
    float gravity_z_upper_;
    int last_button0_ = 0, last_button1_ = 0, last_button2_ = 0, last_button3_ = 0, last_button4_ = 0;
    std::shared_ptr<MotionLoader> motion_loader_;
    size_t motion_frame_ = 0;
    std::vector<const char *> input_names_raw_, output_names_raw_;
    std::unique_ptr<ModelContext> normal_ctx_, motion_ctx_;
    ModelContext* active_ctx_;
    std::shared_ptr<IMUDriver> imu_;
    std::shared_ptr<Decouple> ankle_decouple_;
    std::vector<std::shared_ptr<MotorDriver>> motors_;
    std::unique_ptr<ThreadPool> thread_pool_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_motors_service_, set_zeros_service_, clear_errors_service_, read_motors_service_, init_motors_service_, deinit_motors_service_, start_inference_service_, stop_inference_service_;

    std::mutex motors_mutex_, joint_mutex_, act_mutex_, perception_mutex_, interrupt_mutex_, cmd_mutex_;
    std::vector<float> obs_, act_, last_act_, perception_obs_, motion_pos_, motion_vel_, joint_obs_, cmd_vel_, quat_, ang_vel_, interrupt_action_, joint_torques_;

    void subs_joy_callback(const std::shared_ptr<sensor_msgs::msg::Joy> msg);
    void subs_cmd_callback(const std::shared_ptr<geometry_msgs::msg::Twist> msg);
    void subs_elevation_callback(const std::shared_ptr<std_msgs::msg::Float32MultiArray> msg);
    void subs_joint_state_callback(const std::shared_ptr<sensor_msgs::msg::JointState> msg);
    void apply_action();
    void inference();
    void reset();
    void load_config();
    void setup_model(std::unique_ptr<ModelContext>& ctx, std::string model_path, int input_size);
    void setup_motors();
    void setup_imu();
    void init_motors();
    void deinit_motors();
    void reset_motors();
    void set_zeros();
    void clear_errors();
    void read_motors();
    void init_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void deinit_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                           std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void reset_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void set_zeros_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void clear_errors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void read_motors_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void start_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void stop_inference_srv(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void publish_joint_states();
    void publish_action();
    void publish_imu();

    void exec_motors_parallel(std::function<void(std::shared_ptr<MotorDriver>&, int)> cmd_func);
    void process_close_chain(float* q_in, float* vel_in, float* tau_in, float* target, bool forward_only);
    
    template <typename T>
    void print_vector(const std::string& name, const std::vector<T>& vec) {
        std::stringstream ss;
        ss << name << ": [";
        for (size_t i = 0; i < vec.size(); ++i) {
            ss << vec[i] << (i == vec.size() - 1 ? "" : ", ");
        }
        ss << "]";
        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }
};