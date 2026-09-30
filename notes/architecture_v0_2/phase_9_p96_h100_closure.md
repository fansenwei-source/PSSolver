# Phase 9 P9.6: independent periodic consumer closure

## Result

P9.6 is complete with classification
`PASS_P9_6_INDEPENDENT_PERIODIC_CONSUMER_INSTALLED_WHEEL_H100`.  The
machine-readable authority is
[phase_9_p96_h100_closure.json](phase_9_p96_h100_closure.json).  It binds the
provider commit `d878abd3caecffb99f2cbf9255963a42f401af11`, the independently
maintained consumer qualification commit
`53a1a0903021789c2edb6d029cb49b85071528c9`, and the final v8 archive.

The installed provider and consumer wheels passed the target PyTorch 2.5.1
CPU compatibility gates.  On H100, the public periodic functional runtime
passed its input mapping, replay, checkpoint-stride, checkpointed/full-history
VJP, finite objective and gradient, memory, negative, fallback, and source
identity gates.  Replay was byte-for-byte identical and the checkpointed to
full-history VJP relative L2 error was zero.

## Gradient adjudication

The earlier pointwise finite-difference and Richardson threshold failures are
preserved as historical diagnostics; they were not relabelled as passes and
their thresholds were not changed.  The final authoritative contract is the
independent-direction, gradient-subtracted, second-order Taylor test.  All
five frozen directions passed with finite residuals and without a stable
first-order tail.  The scientific classification is therefore
`GRADIENT_CONSISTENT_TAYLOR_SUPPORTED`.

## Authorization boundary

This closure authorizes P9.7 planning only.  It does not execute or authorize
the Channel functional implementation, qualify Channel pressure
differentiation, allow batch sizes above one, stabilize the provisional API,
change a production default, or establish a control-science result.  Those
boundaries are frozen by P9.7.0 before any Channel runtime code is changed.
