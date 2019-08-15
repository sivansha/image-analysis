cmake_minimum_required(VERSION 2.8)
project( aia3 )

# use the following if only one opencv version is installed
find_package( OpenCV REQUIRED)
# use the following if multiple opencv versions are installed
# replace OPENCV_VN with the version 
# replace OPENCV_PATH with the corresponding path
# example: find_package( OpenCV 3 REQUIRED PATHS "/opt/opencv3")
#find_package( OpenCV OPENCV_VN REQUIRED PATHS "OPENCV_PATH")

add_executable( aia3
                main.cpp
                Aia3.cpp
)

target_link_libraries( aia3 ${OpenCV_LIBS} )
