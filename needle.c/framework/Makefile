CC = clang
CFLAGS = -std=c11 -O2 -Wall -Wextra -Iinclude
AR = ar
SRCS = src/util.c src/tensor.c src/autograd.c src/optim.c src/dataset.c src/io.c src/nn.c
OBJS = $(SRCS:src/%.c=src/%.o)
LIB = libneedle.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $(OBJS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)
