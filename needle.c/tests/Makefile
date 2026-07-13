CC = clang
CFLAGS = -std=c11 -O2 -Wall -Wextra -I../framework/include
LDFLAGS = -L../framework -lneedle -lm

.PHONY: all test clean test_tensor test_autograd test_dataset test_train_step

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

test: test_tensor test_autograd test_dataset test_train_step
	@echo "ALL TESTS GREEN"

clean:
	rm -f unit/test_tensor unit/test_autograd unit/test_dataset integration/test_train_step
