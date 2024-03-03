executable = scripts/test.sh
getenv = True
error = condor_logs/gpu_test.error
log =  condor_logs/gpu_test.log
notification = always
transfer_executable = false
request_memory = 8*1024
request_GPUs = 1
+Research = True
Queue