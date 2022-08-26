func.func @matmul_custom_schedule(%mm: !pdl.operation) {
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    transform.schedule.replace_op_with_custom_schedule %0 {schedule = "matmul_custom_schedule"}
  }
}
