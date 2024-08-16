# Compiling
echo "Compiling $1.cpp"
g++ -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -o "$1" "$1".cpp

# Running
echo "Running $1"
./"$1"