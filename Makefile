# Compiler
CXX      := g++

# Compilation Flags (modify as needed)
CXXFLAGS := -Wall -Wextra -std=c++17 -O2
LDFLAGS  := -lpthread  # Add linker flags if necessary

# Directories
SRC_DIR  := src
INC_DIR  := include
BUILD    := build
OBJ_DIR  := $(BUILD)/objects
BIN_DIR  := $(BUILD)

# Target executable name
TARGET   := mujica

# Find all .cpp files in src/ and map them to .o files
SRC      := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS  := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DEPENDENCIES := $(OBJECTS:.o=.d)

# Build directory structure
build:
	@mkdir -p $(BUILD)
	@mkdir -p $(OBJ_DIR)

# Default target: Build everything
all: build $(BIN_DIR)/$(TARGET)

# Compile each .cpp file to .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# Link all .o files into the final executable
$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Include dependencies for incremental compilation
-include $(DEPENDENCIES)

# Special build modes
debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O3
release: all

sanitize: CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
sanitize: all

# Cleanup
clean:
	-@rm -rvf $(BUILD)

# Print build info
info:
	@echo "[*] Source files:    $(SRC)"
	@echo "[*] Object files:    $(OBJECTS)"
	@echo "[*] Dependencies:    $(DEPENDENCIES)"
	@echo "[*] Include dir:     $(INC_DIR)"
	@echo "[*] Build dir:       $(BUILD)"
	@echo "[*] Target:          $(TARGET)"