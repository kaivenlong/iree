  iree_cc_binary(
    NAME
      klw-compile
    SRCS
      "klw-compile-main.cc"
    DEPS
      iree::klw::compiler::Tools::klw_compile_lib
    DATA
      ${IREE_LLD_TARGET}
    HOSTONLY
  )


