#==============================================================================
# Target Config
#==============================================================================
ARA=1
HOST=0
GPU=0
CUDNN=0
OPENCV=0
DEBUG=0

SHELL = /usr/bin/env bash
ROOT_DIR := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
#==============================================================================
# Toolchain Selection
#==============================================================================
ifeq ($(ARA), 1) 
APPS_DIR := $(ROOT_DIR)
COMMON_DIR := $(ROOT_DIR)/common
# This will overwrite the ROOT_DIR variable from the included makefile
include $(COMMON_DIR)/runtime.mk
CC = $(RISCV_CC)
OPTS=-Ofast
LDFLAGS= -lm 
COMMON= 
CFLAGS=-nostdlib $(RISCV_FLAGS) $(DEFINES) $(RISCV_WARNINGS)
else
CC=gcc
OBJDUMP=objdump
OPTS=-Ofast
LDFLAGS= -lm 
COMMON= 
CFLAGS=-Wall -Wfatal-errors -g
endif

#==============================================================================
# Toolchain Selection
#==============================================================================
VPATH=./src/
EXEC=simplenet
OBJDIR=./obj/

# OBJ= tiny-malloc.o imagenet.o 
OBJ  =  main.o  iconv2d_tensor8_NCHW.o data.S.o 
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h)


#==============================================================================
# Build Targets
#==============================================================================
all:  obj results $(EXEC)

.PHONY: linker_script
linker_script: $(COMMON_DIR)/script/align_sections.sh $(ROOT_DIR)/../../config/$(config).mk
	chmod +x $(COMMON_DIR)/script/align_sections.sh
	rm -f $(COMMON_DIR)/link.ld && cp $(COMMON_DIR)/arch.link.ld $(COMMON_DIR)/link.ld
	$(COMMON_DIR)/script/align_sections.sh $(nr_lanes) $(COMMON_DIR)/link.ld

ifeq ($(ARA), 1) 
$(EXEC):  $(RUNTIME_LLVM)  $(OBJS) linker_script
	$(CC) -Iinclude $(CFLAGS) -o $@ $(OBJS) $(RUNTIME_LLVM) $(RISCV_LDFLAGS) -T$(CURDIR)/common/link.ld
	$(RISCV_OBJDUMP) $(RISCV_OBJDUMP_FLAGS) -D $@ > $@.dump
	$(RISCV_STRIP) $@ -S --strip-unneeded
else
$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	$(OBJDUMP) -D $@ > $@.dump
endif

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@
$(OBJDIR)%.S.o: %.S $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@



obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)
	rm -rf obj
	rm -rf results

