# AppWorld execution environment

This task is an ordinary user request in one simulated AppWorld. In planning
schemas select worker_kind GENERAL or CODE. Do not select WEB_SEARCH or request
browser/network tools. Keep the existing schema contracts and required fields.
Use PLAN so the Scheduler's Final Review can verify the complete task.

Choose GENERAL for direct lookups or straightforward app operations. It executes
the request and reports its own evidence. Choose CODE for substantial data
processing, reusable scripts, or logic that benefits from independent tests.
CODE retains Code Worker, Code Reviewer, repair, and artifact publication. For
AppWorld scripts use code_task delivery_mode ARTIFACT, specify the script/result
artifact and concrete requirements/tests. A published script alone does not mean
the requested app changes occurred. Return evidence of actual AppWorld execution.

appworld_discover reads only API directories and signatures. appworld_execute
queries or modifies business data against this task's persistent world. Docker
execute still runs local code/tests; it cannot access AppWorld's apis, variables,
or files. Send Python source explicitly to appworld_execute to run it there.
No network or package downloading is needed to access the simulated apps.
Use print to inspect outputs; all API arguments are keyword arguments.
Send documentation-only Python to appworld_discover:
print(apis.api_docs.show_app_descriptions())
print(apis.api_docs.show_api_descriptions(app_name="APP_NAME"))
print(apis.api_docs.show_api_doc(app_name="APP_NAME", api_name="API_NAME"))
Use documented supervisor APIs for simulated account information.
Never inspect hidden evaluator, ground truth, or internal implementation files.

General and Code Worker share this task's world. Code Reviewer receives
appworld_discover plus appworld_verify against the same world and should independently query actual
records and check the user's requirements. Do not repeat mutations for verification
or rely on Worker temporary variables: use documented queries with explicit inputs.
appworld_verify is a verification-purpose Python tool, not an enforced read-only
sandbox. Login/documentation calls may be needed; never repair business data there.
For defects request Code Worker repair using the existing review protocol.
Keep dependent app operations sequential; do not run competing writers in parallel.

Only after all requested work is done, call the documented
apis.supervisor.complete_task with any required answer, then report the evidence.
This completion marker is not the official score and does not bypass Scheduler
Final Review. Scheduler checks the original full request, step evidence, outstanding
requirements and reviewer results; it may replan within the existing limits or end
with an honest failure. The external controller alone invokes official evaluation
after a terminal Scheduler Final Review. No agent can invoke the grader tool.
Keep answers concise and include only necessary results and evidence.
