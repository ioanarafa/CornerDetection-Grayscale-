cmake_minimum_required(VERSION 3.30)
project(PiPredefinit)

set(CMAKE_CXX_STANDARD 20)

# ✅ Include corect (doar până la folderul care conține `opencv2/`)
include_directories("D:/opencv/build/include")

# ✅ Directorul cu fișierele .lib
link_directories("D:/opencv/build/x64/vc16/lib")

set(CMAKE_CXX_STANDARD 20)

add_executable(PiPredefinit main.cpp)

target_link_libraries(PiPredefinit opencv_world4110d)