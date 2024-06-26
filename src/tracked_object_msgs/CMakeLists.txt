project(tracked_object_msgs)

set(MSG_FILES
  "Object.msg"
  "ObjectArray.msg"
)

cmake_minimum_required(VERSION 3.5)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# Apend "msg/" to each file name
set(TEMP_LIST "")
foreach(MSG_FILE ${MSG_FILES})
  list(APPEND TEMP_LIST "msg/${MSG_FILE}")
endforeach()
set(MSG_FILES ${TEMP_LIST})

rosidl_generate_interfaces(${PROJECT_NAME}
  ${MSG_FILES}
  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
    nav_msgs
    std_msgs
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
