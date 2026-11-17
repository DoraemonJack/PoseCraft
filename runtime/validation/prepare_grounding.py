import json,collections,hashlib,pathlib
R=pathlib.Path("/path/to/research300/posecraft_val_reference_correction_20261117")
def read(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
requests=read(R/"unique_parse_requests.jsonl"); tasks=read(R/"validation_prompts.jsonl"); out={}
for f in sorted((R/"parse_batches").glob("batch_*.json")):
 for x in json.loads(f.read_text())["items"]:
  assert x["id"] not in out; out[x["id"]]=x
assert set(out)=={x["id"] for x in requests}
entry_parse={i:out[q["id"]] for q in requests for i in q["entry_ids"]}
assert set(entry_parse)=={x["entry_id"] for x in tasks}
recs=[]; jobs={}; rels=collections.Counter()
for t in tasks:
 p=entry_parse[t["entry_id"]]; refs={r["id"]:r for r in p.get("references",[])}; explicit=[]
 for c in p.get("constraints",[]):
  ty=c.get("type", "other"); evidence=c.get("evidence",""); pol=c.get("polarity","positive"); rels[(ty,pol)]+=1
  for rid in c.get("references",[]):
   if rid not in refs:continue
   r=refs[rid]; phrase=r.get("detector_query","").strip()
   if not phrase:continue
   row=dict(phrase=phrase, relation=ty, quote=evidence, source_phrase=r.get("phrase"), polarity=pol)
   if ty=="blocks_view_of":row["placement_relation_proxy"]="near"
   elif ty=="avoids_blocking_view_of":row["placement_relation_proxy"]="not near"
   elif ty=="far" and pol=="negative":row["placement_relation_proxy"]="near"
   explicit.append(row)
   jobs[(t["scene"],phrase.lower())]=dict(scene=t["scene"],phrase=phrase,detector_phrase=phrase.lower(),kind="explicit",cleanup_status="detector_candidate")
 recs.append(dict(**t,original_entry_id=t["entry_id"],explicit=explicit,inferred=[],parse=p))
(R/"mapped_val_tasks.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in recs))
(R/"missing_jobs.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in jobs.values()))
print(json.dumps(dict(tasks=len(recs),unique_queries=len(jobs),relations=[[*k,v] for k,v in rels.most_common()]),ensure_ascii=False))
