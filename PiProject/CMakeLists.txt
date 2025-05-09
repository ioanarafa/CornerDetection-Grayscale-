cmake_minimum_required(VERSION 3.10)
project(PiProject)

set(CMAKE_CXX_STANDARD 17)

# ✅ Include corect (doar până la folderul care conține `opencv2/`)
include_directories("D:/opencv/build/include")

# ✅ Directorul cu fișierele .lib
link_directories("D:/opencv/build/x64/vc16/lib")

add_executable(PiProject main.cpp)

# ✅ Link explicit cu fișierul .lib de debug
target_link_libraries(PiProject opencv_world4110d)
