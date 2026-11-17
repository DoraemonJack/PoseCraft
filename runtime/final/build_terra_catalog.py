import pathlib,importlib.util,json,collections,hashlib,traceback,os
R=pathlib.Path('/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1');O=pathlib.Path('/path/to/research300/posecraft_terra_parse_epoch8_20261116');old=R/'references'; new=O/'detector';status=O/'catalog_status.json'
def save(v): status.write_text(json.dumps(v,indent=2))
def rows(p):
 with open(p,encoding='utf8') as f:
  for line in f:yield json.loads(line)
try:
 assert json.loads((O/'association_status.json').read_text())['stage']=='complete'
 source=(R/'build_test_reference_catalog.py').read_text();source=source.replace("OUT = ROOT / 'references'", "OUT = Path('/path/to/research300/posecraft_terra_parse_epoch8_20261116/catalog')")
 source=source.replace("relation_type = relation(query.get('relation'))", "negative_proxy = query.get('placement_relation_proxy') == 'not near'\n                    relation_type = 'near' if query.get('placement_relation_proxy') in ('near', 'not near') else relation(query.get('relation'))")
 source=source.replace("query_specs.append(dict(type=relation_type, phrase=phrase, candidate_keys=sorted(set(keys))))", "query_specs.append(dict(type=relation_type, phrase=phrase, negative_proxy=negative_proxy, candidate_keys=sorted(set(keys))))")
 source=source.replace('compact_queries = []\n            if query_specs', 'compact_queries = []\n            negative_queries = []\n            if query_specs')
 original="""                    compact_queries.append(dict(type=query['type'], phrase=query['phrase'], candidate_indices=sorted(allowed),
                                                actual_grounder_logits=values.tolist(), effective_logits=effective,
                                                relation_mask=bool(allowed and not null_top), null_is_top=null_top,
                                                reference_kind='explicit'))"""
 modified="""                    record_query = dict(type=query['type'], phrase=query['phrase'], candidate_indices=sorted(allowed),
                                                actual_grounder_logits=values.tolist(), effective_logits=effective,
                                                relation_mask=bool(allowed and not null_top), null_is_top=null_top,
                                                reference_kind='explicit')
                    (negative_queries if query['negative_proxy'] else compact_queries).append(record_query)"""
 assert original in source;source=source.replace(original,modified)
 source=source.replace("for explicit in compact_queries:","for explicit in compact_queries + negative_queries:")
 source=source.replace("reference_mask=[True] * len(compact_candidates), queries=compact_queries, unresolved=unresolved,", "reference_mask=[True] * len(compact_candidates), queries=compact_queries, negative_queries=negative_queries, unresolved=unresolved,")
 ns={'__name__':'terra_catalog_builder','__file__':str(R/'build_test_reference_catalog.py')};exec(compile(source,str(R/'build_test_reference_catalog.py'),'exec'),ns);out=ns['OUT'];out.mkdir(exist_ok=True)
 src=O/'corrected_with_relation_proxies.jsonl';(out/'mapped_test_tasks.jsonl').write_bytes(src.read_bytes())
 feat={}
 for p in (old/'grounder_candidate_features_v1.jsonl',new/'grounder_candidate_features_v1.jsonl'):
  for x in rows(p):
   k=(x['scene'],x['detector_phrase'].strip().lower());v=[c for c in x['candidates'] if c.get('status')=='available']
   if v or k not in feat:feat[k]=v
 lookup=collections.defaultdict(set)
 for x in rows(old/'cleaned_detection_jobs_v2.jsonl'):
  if x['cleanup_status']=='detector_candidate':lookup[(x['scene'],x['phrase'].strip().lower(),x['kind'])].add(x['detector_phrase'].lower())
 for x in rows(old/'split_between_candidates_v1.jsonl'):lookup[(x['scene'],x['parent_phrase'].strip().lower(),x['kind'])].add(x['detector_phrase'].lower())
 for scene,phrase in feat:lookup[(scene,phrase,'explicit')].add(phrase)
 ns['torch'].set_num_threads(2);save(dict(stage='loading_models',cached_groups=len(feat)));models,splits,vocab,ident=ns['models_and_splits']();save(dict(stage='scoring',cached_groups=len(feat)))
 path,counts=ns['score'](feat,lookup,models,splits,vocab);neg=0;neg_active=0
 for z in rows(path):
  for q in z.get('negative_queries',[]):
   neg+=1;neg_active+=int(q['relation_mask'])
 manifest=dict(status='complete',data_path=str(path),data_sha256=ns['sha'](path),input_sha256=ns['sha'](src),task_count=3500,counts=counts,negative_queries=neg,negative_active=neg_active,negative_rule='not-near represented in separate negative_queries, near vocabulary used only for selecting reference',features='original cached plus newly grounded observations; no GT objects',grounder_identities=ident)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));save(manifest)
except BaseException as e:save(dict(stage='failed',error=repr(e),traceback=traceback.format_exc()));raise
