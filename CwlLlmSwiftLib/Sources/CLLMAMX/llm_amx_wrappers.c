#include "LLMAMXExports.h"

#if defined(__APPLE__) && defined(__aarch64__)
#define AMX_NOP_OP_IMM5(op, imm5) \
    __asm__("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
    __asm__(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define AMX_LDX(gpr) AMX_OP_GPR(0, gpr)
#define AMX_LDY(gpr) AMX_OP_GPR(1, gpr)
#define AMX_LDZ(gpr) AMX_OP_GPR(4, gpr)
#define AMX_STZ(gpr) AMX_OP_GPR(5, gpr)
#define AMX_SET() AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR() AMX_NOP_OP_IMM5(17, 1)
#define AMX_MATFP(gpr) AMX_OP_GPR(21, gpr)
#endif

uint8_t amx_is_available(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    return 1;
#else
    return 0;
#endif
}

void amx_set(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    AMX_SET();
#endif
}

void amx_clr(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    AMX_CLR();
#endif
}

void amx_ldx(uint64_t operand) {
#if defined(__APPLE__) && defined(__aarch64__)
    AMX_LDX(operand);
#else
    (void)operand;
#endif
}

void amx_ldy(uint64_t operand) {
#if defined(__APPLE__) && defined(__aarch64__)
    AMX_LDY(operand);
#else
    (void)operand;
#endif
}

void amx_ldz(uint64_t operand) {
#if defined(__APPLE__) && defined(__aarch64__)
    AMX_LDZ(operand);
#else
    (void)operand;
#endif
}

void amx_stz(uint64_t operand) {
#if defined(__APPLE__) && defined(__aarch64__)
    AMX_STZ(operand);
#else
    (void)operand;
#endif
}

void amx_matfp(uint64_t operand) {
#if defined(__APPLE__) && defined(__aarch64__)
    AMX_MATFP(operand);
#else
    (void)operand;
#endif
}
