#!/bin/bash
set -e

# Get the script's directory (this is the key fix)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"  # Parent directory of script directory

# Create the output directory if it doesn't exist
mkdir -p "$PROJECT_DIR/src/generated"

# Generate Python code from proto definition
python3 -m grpc_tools.protoc \
  -I "$PROJECT_DIR/proto" \
  --python_out="$PROJECT_DIR/src/generated" \
  --grpc_python_out="$PROJECT_DIR/src/generated" \
  "$PROJECT_DIR/proto/processor.proto"

# Create __init__.py to make the directory a proper package
touch "$PROJECT_DIR/src/generated/__init__.py"

# Fix the import in the generated _grpc.py file
# For macOS compatibility, we need to use a different sed syntax
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS version (BSD sed)
  sed -i '' 's/import processor_pb2/from generated import processor_pb2/' \
    "$PROJECT_DIR/src/generated/processor_pb2_grpc.py"
else
  # Linux version (GNU sed)
  sed -i 's/import processor_pb2/from generated import processor_pb2/' \
    "$PROJECT_DIR/src/generated/processor_pb2_grpc.py"
fi

echo "Proto compilation complete!"