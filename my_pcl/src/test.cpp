#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversations/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

ros::Publisher pub;

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    sensor_msgs::PointCloud2 output; //buffer for the data

    output = *input; //the data is processing here

    pub.publish (output);
}

int main (int argc, char** argv)
{
    ros::init (argc, argv, "Hello, PCL");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe ("input", 1, cloud_cb);

    pub = nh.advertise<sensor_msgs::PointCLoud2> ("output", 1);

    ros::spin(); 
}