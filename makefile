all: build

build: summary
	gitbook build

install:
	pip3 install torch torchvision torchaudio

installgpu:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

.PHONY: clean build all