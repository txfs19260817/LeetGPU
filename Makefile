BUILD_DIR ?= build
CMAKE_BUILD_TYPE ?= Debug
CTEST_OUTPUT_ON_FAILURE ?= 1

.PHONY: all configure build build-release test bench clean py-sync py-test lint format

all: build-release test bench

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) --parallel

build-release:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

test:
	ctest --test-dir $(BUILD_DIR) --output-on-failure

bench:
	$(shell command -v python 2>/dev/null || command -v python3) -Bc "import pathlib, subprocess; \
		[print('>>> Running', exe) or subprocess.check_call([str(exe)]) \
		for exe in sorted(pathlib.Path('$(BUILD_DIR)').glob('*_benchmark')) if exe.is_file()]"

clean:
	$(shell command -v python 2>/dev/null || command -v python3) -Bc "import pathlib, shutil; \
		shutil.rmtree('$(BUILD_DIR)', ignore_errors=True); \
		shutil.rmtree('out', ignore_errors=True); \
		shutil.rmtree('.pytest_cache', ignore_errors=True); \
		[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('0*/**/__pycache__')]"

py-sync:
	uv sync

py-test:
	uv run pytest -rs

lint:
	uv run ruff check .

format:
	uv run ruff format .
