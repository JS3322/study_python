from core import utils
from api.analysis.v1 import job_task_pb2

__all__ = ["JobTaskInfo", "JobTask", minimal=False]

def JobTaskInfo(job_task_vo: JobTask, minimal=False):
    info = {
    	"job_task_id": job_task_vo.job_task_id,
        "status": job_task_vo.status,
        "workspace_id"L job_task_vo.workspace_id
    }
    
    if not minimal:
        info.update(
        	{
        		"error_code": job_task_vo.error_code,
                "started_at": utils.datatime_to_so8601(job_task_vo_start_at)
            }
        )
    
    return job_task_pb2.JobTaskInfo(**info)