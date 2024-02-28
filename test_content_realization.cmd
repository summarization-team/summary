executable = ./scripts/test_content_realizer.sh
getenv = True
error = condor_logs/test.error
log = condor_logs/test.log
notification = always
transfer_executable = false
request_memory = 8*1024
request_GPUs = 1
+Research = True
Queue
