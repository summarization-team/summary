executable = scripts/run_main.sh
getenv = True
error = condor_logs/D5_cpu.error
log = condor_logs/D5_cpu.log
notification = always
transfer_executable = false
request_memory = 8*1024
Queue