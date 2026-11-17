import pathlib,importlib.util,json,collections,traceback
R=pathlib.Path("/path/to/research300/posecraft_val_reference_correction_20261117"); T=pathlib.Path("/path/to/research300/posecraft_terra_parse_epoch8_20261116"); BASE=pathlib.Path("/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1");OUT=R/"catalog";OUT.mkdir(exist_ok=True)
def rows(p):
 with open(p,encoding="utf8") as f:
  for s in f:
   if s.strip():yield json.loads(s)
def save(v): (R/"catalog_status.json").write_text(json.dumps(v,indent=2))
try:
 tasks=list(rows(R/"mapped_val_tasks.jsonl"))
 for t in tasks:
  for q in t["explicit"]:
   if q["relation"]=="other":q["relation"]=q["quote"]
   if q["relation"]=="not_visible_from" or (q["relation"]=="visible_from" and q.get("polarity")=="negative"):q["relation"]="not visible"
 (OUT/"mapped_val_tasks.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in tasks))
 source=(BASE/"build_test_reference_catalog.py").read_text()
 source=source.replace("OUT = ROOT / 'references'", "OUT = Path('"+str(OUT)+"')")
 source=source.replace("OUT / 'mapped_test_tasks.jsonl'", "OUT / 'mapped_val_tasks.jsonl'")
 source=source.replace("ROOT / 'text_features.npy'", "Path('"+str(R/"text_features.npy")+"')")
 source=source.replace("3500", "2566")
 source=source.replace("relation_type = relation(query.get('relation'))", "negative_proxy = query.get('placement_relation_proxy') == 'not near'\n                    relation_type = 'near' if query.get('placement_relation_proxy') in ('near', 'not near') else relation(query.get('relation'))")
 source=source.replace("query_specs.append(dict(type=relation_type, phrase=phrase, candidate_keys=sorted(set(keys))))", "query_specs.append(dict(type=relation_type, phrase=phrase, negative_proxy=negative_proxy, candidate_keys=sorted(set(keys))))")
 source=source.replace("compact_queries = []\n            if query_specs", "compact_queries = []\n            negative_queries = []\n            if query_specs")
 old="""                    compact_queries.append(dict(type=query['type'], phrase=query['phrase'], candidate_indices=sorted(allowed),
                                                actual_grounder_logits=values.tolist(), effective_logits=effective,
                                                relation_mask=bool(allowed and not null_top), null_is_top=null_top,
                                                reference_kind='explicit'))"""
 new="""                    record_query = dict(type=query['type'], phrase=query['phrase'], candidate_indices=sorted(allowed),
                                                actual_grounder_logits=values.tolist(), effective_logits=effective,
                                                relation_mask=bool(allowed and not null_top), null_is_top=null_top,
                                                reference_kind='explicit')
                    (negative_queries if query['negative_proxy'] else compact_queries).append(record_query)"""
 assert old in source;source=source.replace(old,new).replace("for explicit in compact_queries:","for explicit in compact_queries + negative_queries:")
 source=source.replace("reference_mask=[True] * len(compact_candidates), queries=compact_queries, unresolved=unresolved,", "reference_mask=[True] * len(compact_candidates), queries=compact_queries, negative_queries=negative_queries, unresolved=unresolved,")
 ns={"__name__":"val_catalog_builder","__file__":str(BASE/"build_test_reference_catalog.py")};exec(compile(source,ns["__file__"],"exec"),ns);ns["torch"].set_num_threads(2)
 feat={}
 for p in [BASE/"references/grounder_candidate_features_v1.jsonl",T/"detector/grounder_candidate_features_v1.jsonl",R/"detector/grounder_candidate_features_v1.jsonl"]:
  for x in rows(p):
   k=(x["scene"],x["detector_phrase"].strip().lower());v=[c for c in x["candidates"] if c.get("status")=="available"]
   if v or k not in feat:feat[k]=v
 lookup=collections.defaultdict(set)
 for t in tasks:
  for x in t["explicit"]:
   phrase=x["phrase"].strip().lower();lookup[t["scene"],phrase,"explicit"].add(phrase)
 models,splits,vocab,ident=ns["models_and_splits"]();path,counts=ns["score"](feat,lookup,models,splits,vocab)
 records=list(rows(path));manifest=dict(status="complete",split="val",data_path=str(path),data_sha256=ns["sha"](path),task_manifest_sha256=ns["sha"](R/"tasks_manifest.json"),task_count=len(records),counts=counts,negative_queries=sum(len(x.get("negative_queries",[])) for x in records),grounder_identities=ident)
 (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2));save(manifest);print(json.dumps(manifest),flush=True)
except BaseException as e:save(dict(status="failed",error=repr(e),traceback=traceback.format_exc()));raise
