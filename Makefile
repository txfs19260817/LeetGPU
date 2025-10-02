BUILD_DIR=build

.PHONY: build test bench clean py-test py-clean py-sync

all: build-release test bench

build:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake .. && make -j

build-release:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j

test:
	cd $(BUILD_DIR) && ctest --output-on-failure

bench:
	cd $(BUILD_DIR) && for exe in $$(find . -maxdepth 1 -type f -executable -name "*_benchmark"); do \
	    echo ">>> Running $$exe"; \
	    $$exe; \
	done

clean:
	rm -rf $(BUILD_DIR)

py-sync:
	uv sync

py-test:
	uv run pytest -rs