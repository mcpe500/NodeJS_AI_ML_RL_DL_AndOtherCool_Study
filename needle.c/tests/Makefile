CC = clang
CFLAGS = -std=c11 -O2 -Wall -Wextra -I../framework/include
LDFLAGS = -L../framework -lneedle -lm

.PHONY: all test clean test_tensor test_autograd test_dataset test_train_step test_bpe test_fc_bin test_generate

all: test

../framework/libneedle.a:
	$(MAKE) -C ../framework

test_tensor: unit/test_tensor.c ../framework/libneedle.a
	$(CC) $(CFLAGS) unit/test_tensor.c $(LDFLAGS) -o unit/test_tensor
	./unit/test_tensor

test_autograd: unit/test_autograd.c ../framework/libneedle.a
	$(CC) $(CFLAGS) unit/test_autograd.c $(LDFLAGS) -o unit/test_autograd
	./unit/test_autograd

test_dataset: unit/test_dataset.c ../framework/libneedle.a
	$(CC) $(CFLAGS) unit/test_dataset.c $(LDFLAGS) -o unit/test_dataset
	./unit/test_dataset

test_train_step: integration/test_train_step.c ../framework/libneedle.a
	$(CC) $(CFLAGS) integration/test_train_step.c $(LDFLAGS) -o integration/test_train_step
	./integration/test_train_step

test_bpe: unit/test_bpe.c ../framework/libneedle.a
	$(CC) $(CFLAGS) unit/test_bpe.c $(LDFLAGS) -o unit/test_bpe
	./unit/test_bpe

test_fc_bin: unit/test_fc_bin.c ../framework/libneedle.a
	$(CC) $(CFLAGS) unit/test_fc_bin.c $(LDFLAGS) -o unit/test_fc_bin
	./unit/test_fc_bin

test_generate: unit/test_generate.c ../framework/libneedle.a
	$(CC) $(CFLAGS) unit/test_generate.c $(LDFLAGS) -o unit/test_generate
	./unit/test_generate

test: test_tensor test_autograd test_dataset test_train_step test_bpe test_fc_bin test_generate
	@echo "ALL TESTS GREEN"

clean:
	rm -f unit/test_tensor unit/test_autograd unit/test_dataset integration/test_train_step unit/test_bpe unit/test_fc_bin unit/test_generate
