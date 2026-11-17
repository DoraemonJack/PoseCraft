"""Read-only check of the prepared official command and input identity."""
import json
import shlex
import watch_official as w

client = w.official_connection()
try:
    code = ('import sys,json; from pathlib import Path; sys.path.insert(0,'+repr(w.DEST)+'); '
            'import evaluate_test300_official as e; '
            'pred=e.ROOT/"official_batched_3process_45g_v1/batch_01/predictions.json"; '
            'e.validate_predictions(pred); '
            'cmd,native,dependencies=e.construct_command(e.read(e.TEMPLATE),pred,'
            'Path('+repr(w.DEST)+')/"preflight_unused_output","orientation-preflight-only"); '
            'image=json.loads(e.docker("image","inspect",e.IMAGE).stdout); assert image[0]["Id"]==e.IMAGE; '
            'print(json.dumps(dict(status="passed",tasks=300,dependencies=len(dependencies),image=e.IMAGE,native=str(native))))')
    result = w.command(client, 'python3 -c '+shlex.quote(code))
    data = json.loads(result)
    w.write(w.HERE/'official_preflight.json', data)
    print(result)
finally:
    client.close()
