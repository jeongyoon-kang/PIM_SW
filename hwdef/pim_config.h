// pim_config.h — compiled-in defaults for every platform constant (ch2, the
// version1.0 image).
//
// scripts/setup.sh passes the selected conf as -D (hwdef/gen_config.sh --defs), and
// each value below applies only where no -D was given.  A build that took these
// defaults has PIM_CONFIG_FROM_CONF 0, and pim_platform_check() refuses to run it:
// build through setup.sh.
//
// To refresh after editing platform/ch2.conf, replace the block below with the
// output of `hwdef/gen_config.sh --header` while platform/active points at ch2.
#ifndef PIM_CONFIG_H
#define PIM_CONFIG_H

#ifndef PIM_CONFIG_FROM_CONF
#define PIM_CONFIG_FROM_CONF 0
#endif
#ifndef PIM_PLATFORM_NAME
#define PIM_PLATFORM_NAME   "ch2"
#endif
#ifndef PIM_CONF_PATH
#define PIM_CONF_PATH       ""
#endif
#ifndef PIM_NCH
#define PIM_NCH             2u
#endif
#ifndef PIM_NBANK
#define PIM_NBANK           16u
#endif
#ifndef PIM_BAR2_AXI_BASE
#define PIM_BAR2_AXI_BASE   0x020200000000ULL
#endif
#ifndef PIM_OFF_GPR
#define PIM_OFF_GPR         0x000000ULL
#endif
#ifndef PIM_OFF_CFR
#define PIM_OFF_CFR         0x400000ULL
#endif
#ifndef PIM_OFF_VIOL
#define PIM_OFF_VIOL        0x401000ULL
#endif
#ifndef PIM_OFF_IMEM
#define PIM_OFF_IMEM        0x600000ULL
#endif
#ifndef PIM_HBM_BASE
#define PIM_HBM_BASE        0x004000000000ULL
#endif
#ifndef PIM_HBM_CH_SPAN
#define PIM_HBM_CH_SPAN     0x400000000ULL
#endif
#ifndef PIM_BANK_STRIDE
#define PIM_BANK_STRIDE     0x40000000ULL
#endif
#ifndef PIM_BANK_WINDOW
#define PIM_BANK_WINDOW     0x10000000ULL
#endif
#ifndef PIM_MC_BASE
#define PIM_MC_BASE         0x020400000000ULL
#endif
#ifndef PIM_MC_CH_SPAN
#define PIM_MC_CH_SPAN      0x100000000ULL
#endif
#ifndef PIM_UPLOAD_PATH
#define PIM_UPLOAD_PATH     "direct"
#endif
#ifndef PIM_NTAIL_POLICY
#define PIM_NTAIL_POLICY    "pad"
#endif
#ifndef PIM_SCHEDULE
#define PIM_SCHEDULE        "group"
#endif
#ifndef PIM_FEATURE_T_LATCH
#define PIM_FEATURE_T_LATCH 0
#endif
#ifndef PIM_ADDR_MAP
#define PIM_ADDR_MAP        1
#endif

#endif // PIM_CONFIG_H
