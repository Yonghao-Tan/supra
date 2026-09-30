PYTHON ?= python3
RUN_ID ?= public-check
JOBS ?= 8
.DEFAULT_GOAL := unit

.PHONY: check unit controls help

check: unit controls

unit:
	$(PYTHON) -B scripts/run_unit_tests.py --run-id $(RUN_ID)-unit --jobs $(JOBS)

controls:
	$(PYTHON) -B scripts/run_unit_tests.py --run-id $(RUN_ID)-softmax --case softmax --case matmul-activation --case matmul-output --case elementwise --case attention-context-writer --jobs $(JOBS)
	$(PYTHON) -B scripts/run_unit_tests.py --run-id $(RUN_ID)-stream --case stream --jobs $(JOBS)
	$(PYTHON) -B scripts/run_unit_tests.py --run-id $(RUN_ID)-memory-control --case hidden-transfer --case axi-master --case ddr-model --case token-loader --case dependency --case invalidation --case refresh-score --jobs $(JOBS)
	$(PYTHON) -B scripts/run_unit_tests.py --run-id $(RUN_ID)-psme --case psme --jobs $(JOBS)
	$(PYTHON) -B scripts/run_unit_tests.py --run-id $(RUN_ID)-atse --case atse --jobs $(JOBS)
	$(PYTHON) -B scripts/run_unit_tests.py --run-id $(RUN_ID)-sequencer --case sequencer --jobs $(JOBS)

help:
	@printf '%s\n' \
		'make check    Run all public model-free unit checks' \
		'make unit     Run numeric, PE, quantizer, SRAM, layer-loader and raw-checker checks (default)' \
		'make controls Run Softmax, Matmul activation/output, stream, hidden transfer, AXI/DDR, token/dependency, PSME, ATSE and sequencer checks'
