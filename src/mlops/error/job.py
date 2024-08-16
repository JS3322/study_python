
class ERROR_JOB_STATE(ERROR_UNKNOWN):
    _message = 'Only running jobs can be canceld. (job_state = {job_state})'
    
class ERROR_DUPLICATE_JOB(ERROR_UNKNOWN):
    _message = 'The same job is already running. (data_source_id = {data_source_id})'
    
class ERROR_GET_JOB_TASKS(ERROR_UNKOWN):
    _message = 'Faild to get job tasks. (secret_id={secret_id}, data_source_id={data_source_id}, reason={reason})'