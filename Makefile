BUILD_DIR ?= build

.PHONY: build test bench clean py-test py-clean py-sync

all: build-release test bench

build:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake -DNVBench_ENABLE_TESTING=OFF .. && make -j

build-release:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release -DNVBench_ENABLE_TESTING=OFF .. && make -j

test:
	cd $(BUILD_DIR) && ctest --output-on-failure

bench:
	cd $(BUILD_DIR) && for exe in $$(find . -maxdepth 1 -type f -executable -name "*_benchmark"); do \
	    echo ">>> Running $$exe"; \
	    $$exe; \
	done

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