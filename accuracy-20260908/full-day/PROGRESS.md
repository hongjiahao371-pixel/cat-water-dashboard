# 全天验证跟进状态

更新时间：2026-09-09 05时，时区 Asia/Shanghai。用户明确授权持续跟踪直至验证完成。

## 最新推进（03/05时跟进合并）

- 正式ROI仍false，legacy运行正常（本轮初核约4.8FPS），没有修改生产判定、历史或通知。
- **画面映射有昼夜实证**：9/7白天原录像2560×1440与实时事件960×720，同视野直接缩放后SIFT背景292内点，直接位置偏差中位0.64px；9/6夜间喷泉区域46内点、0.89px。支持本开发场景回放显式resize960×720，不是任意假设。geometry/mapping.json、night-mapping.json、原片及截图本地已保留。仍有编码质量差异，不声称逐像素等同实时。
- **时间不能按事件ID配对**：白天event-20260907-143208视频首帧OSD14:30:03；夜间review-20260906-033614-715417截图OSD03:33:22。记录时间包含延迟/预缓存，须用原片+PTS建立区间，不按ID时间冒充实际喝水。原因尚未完全定位，不断言全由某个缓存导致。
- **独立夜间候选**：盲查导航页40、41、42定位03:32–03:36附近一次到访。原始四片（33230/33330/33431/33532）已下载到reference-night/，共约41MB；240张1fps密集导航图已全部视觉检查，reference-draft.json保留可能饮水/抬头暂停/IR切换黑暗区间，明确PROVISIONAL，不是最终真值，不可据此算全天召回。
- 可见约36.9–49.2秒（第一片）经历红外/彩色/过曝切换，无法直接确认该段饮水；后段有数次暂停。下一步须原片更密集核边界，保留歧义，不用模型预测补标签。
- **冻结运行配置**：NAS和本地frozen-runtime-20260909/包含settings、calibration（600权限，不打印/公开）、26个非敏感配置及模型/代码指纹。replay_realtime.py新增--runtime-env，避免只依赖默认参数。跨文件3项测试仍通过。
- **已启动cat-water-night-replay**：断网、1CPU、1536MB，原片/源项目只读；用已证实的960×720、固定5fps、冻结配置顺序跑legacy与roi，输出reference-night/sealed-{mode}.json/log。独立参考区间定稿前**不要读取这些预测内容**，只检查容器状态和概括的完成日志。该任务仅四分钟开发片段，不是全天验收；不读预测不妨碍先跑计算。
- 所有此前索引均已完成，勿重新生成。下一轮先检查night-replay状态、完善独立参考边界，再解封这段预测排错；并继续复核其余全天导航/原片，不可只挑模型报出的片段。

## 生产与自动跟进

- 正式服务已回到 legacy_cat_crop，models/roi_drinking.json enabled=false。候选模型和接入代码保留，禁止仅凭此前抽样结果重新激活。
- 当前任务内 heartbeat 自动跟进已创建，ID `automation`，每两小时运行。仅实质进展、失败、需用户动作或完成时通知；完成报告后暂停，不归档任务。
- 本地工作区 `/Users/js/cat-water-accuracy-20260908`；NAS `/volume1/docker/cat-water-dashboard/accuracy-20260908`。
- Paramiko 持久连接 exec session 17404（若仍存在）通过 write_stdin 接收单行 JSON 的 cmd/get/put。脚本 `ssh_session.py`。Mac PTY 单行有长度限制：长远程脚本必须先 put 文件再执行，不要发送超过约1000字节的 JSON 行。密码只用当前任务已有授权，不写文件或自动任务提示；新会话凭据需安全交互输入。

## 已完成的连续录像检查

- `cat-water-day-coverage` 已结束，exit=0。9月6日1419片段，另读9月5日最后1片段覆盖跨日边界，共1420文件。
- coverage.json 已同步本地。元数据探测错误0，按文件名时间+容器时长覆盖86299.944秒（99.884%），累计约100.056秒细碎间隙，无单个>2秒缺口。
- 这不是完整解码通过，也不是饮水真值。摄像机OSD与文件时钟偏移尚未校准，历史事件ID曾错位，不能按历史关联标签当真值。

## 冻结评估日期

- 采用连续7个自然日窗口：2026-09-05至2026-09-11。
- 开发/排错日期：9月5–8日。先用9月6日建立全流程。
- 独立前瞻验收日期：9月9、10、11日。尚未完整产生，不使用这些日期图像训练或调阈值；保留至少三整天，最早9月12日录像齐全后才能完成最终验收，标注可能需要更久。
- 冻结的是日期划分，不代表数据已经逐帧保存或核对。后续检查录像保留期，避免验收源片段被覆盖。
- 旧模型YOLOv8n+原猫框分类器；候选ROI SHA256 `42ca205f089d0121ed946eabb4a8331458a0c47365cdc5cfbd31bd38549a8b85`。正式disabled配置与stage/models/roi_drinking.json启用的离线配置分开。

## 已完成索引（本轮更新）

- 9月9日首次heartbeat核对：正式ROI false，legacy_cat_crop运行正常，夜间约3.2FPS；未修改生产代码、配置或历史。
- blind-index已全部完成：284页、8064缩略图、生成错误0。完整74MB索引已下载解包到本地full-day/review-index-20260906/，不必重下载或重跑。两个离线容器均已结束；后续不是等待容器，而是推进复核和回放。
- 只查看了导航页12、24、36（此前页0也看过），均为夜间空碗缩略图。navigation-review.json明确只记录导航观察，不给整个时间段贴非饮水真值。284页并未全部人工复核，更没有全天完整视频真值。
- NAS当前录像目录覆盖20260809–20260909（32天），剩余约6TB；近期验证窗口暂有保留空间，但不能据目录推定固定留存政策。

## 实时回放工具（本轮新增）

- full-day/replay_realtime.py直接调用detection_worker.main，不重新实现一套近似门控。仅替换视频输入、虚拟时钟、落盘/通知输出；网络禁止，输出仅本地报告。包含运动预筛、分类器、持续判定和饮水结束。
- 跨录像连续读取；明显缺口模拟断流，重叠时间戳不重复送入。test_replay_capture.py三项测试通过。
- 在此前已复核的9月7日短视频上双版冒烟：各91帧，两版均完成start/end，输出smoke-legacy/roi.json及日志。这是工具测试，不能当作全天结果。
- 协议仍需确认：当前固定5fps来源时间戳，不模拟CPU抖动；支持显式resize，但不能随意把原始16:9录像拉伸为已录实时事件的4:3。必须查清历史录像/实时输入视场与几何映射后再做正式对比。
- 已只读取非敏感生产参数：detector NIGHT_HIRES_ENABLED=false，与回放关闭hires一致；QSV SCALE_WIDTH=960、OUTPUT_FPS=8；生产主循环实际采样率与输出流8fps不同。正式回放还需使用当前冻结settings和calibration，不能把原图直接输入当作已完全复现实时输入。
- 工具冒烟出现候选active_drink_sec 15.6大于event duration14.6，可能是前置锚点计入但事件起点较晚；记为开发集时长定义待查，不自行从指标中隐藏，也不在本轮改正式逻辑。

## 索引构建方法（历史记录）

- `cat-water-blind-index`：断网、0.75 CPU、512MB，源录像只读，输出 full-day/review-index-20260906/。
- 脚本 build_review_index.py，每5个约一分钟录像生成一页均匀10秒导航缩略图，不显示任何模型预测；pages.jsonl逐页落盘，可按已完成page恢复。预计284页。
- **索引仅用于导航，绝不能把10秒抽帧当作全天真值或认定没有短时饮水。** 真实标签仍需查看原始连续片段，并复核模型未触发时段；source+approx_offset只能帮助定位，精确起止需原视频。

## 下一次执行

1. 核对生产enabled=false、服务正常。索引已完成，不再重复等候/生成。优先解决回放输入几何映射，冻结当前settings/calibration及完整非敏感检测参数。
2. 下载少量已完成索引，查看场景和空碗基准，建立独立复核记录。不要先看模型预测；可疑片段必须原帧率连续画面复核，标明不确定性与未审核时段。
3. 为9月6日开发集建立实际实时状态机回放工具（沿用production main的门控语义、运动预筛、时序/采样率），断网且不写事件。独立参考区间完成后才计算模型指标；事后archive结果单列。
4. 逐步覆盖9月5–8日开发集并处理错误；9月9–11日只作最终验收。若人工真值无法可靠完成，应提供精简证据请求用户协助，而非把模型输出当真值或反复空转。
5. 依据ACCEPTANCE.md交付可通过或不通过的完整报告。当前没有全天准确率结论，不承诺新版一定提升，不因日期达到就自动声称验收完成。

### 本轮继续：夜间参考边界复核
已盲审9张局部放大边界图（请求间隔0.25秒，实际PTS含重复帧），结合此前4张1秒图，冻结 `reference-night/reference-frozen.json`。可见伸舌支持真实舔水；抬头暂停、初次靠近不确定及红外暗画面/过曝单列。仅为一个开发样本的带不确定性参考，不是逐帧穷尽标注或全天真值。尚未打开两版 sealed 预测结果。后续可进行该开发到访的诊断对比，但不得推断全天准确率。

## 2026-09-09 07:00 follow-up
Night replay exited0. Unsealed after frozen reference. Both detected one visit; legacy active1.0s, ROI65.4s, ROI trigger19.552s earlier. Later clear lapping still missed. See reference-night/diagnosis.md. Actual cadence948/242.672=3.91fps, not strictly5fps. Preparing cat-water-day-replay-20260906 with1420 source files including previous-day boundary. Outputs sealed under replay-day-20260906; no full-day scoring before independent reference completion. Next check job and continue blind footage review; do not repeat night replay.

Day job launched successfully: cat-water-day-replay-20260906, container bf7ad6198224420619f99d7be77e8878505f86ec22edd05119f321894928957c. Resource limits1CPU/1536MB, network none, source read-only.

Initial day launch exited1 due manifest list vs required rows wrapper; corrected manifest before any inference and restarted same container. Production checked ROIfalse, oktrue, legacy_cat_crop,5.4FPS.

## 2026-09-09 09:00 follow-up
Full-day job exited137/OOM at23:00UTC. Diagnosed MagicMock evidence recorder retaining frame arguments; replaced with stateless NullEvidenceRecorder. Weakref regression proves frames released; all4 replay tests pass. No detector logic/model changes. Uploaded replay tool and restarted same container; added prediction-free progress JSON every60 source seconds. Next check memory/progress before further restart, do not change memory cap blindly. Blind navigation pages43,44,45 reviewed: empty fountain in displayed frames, no full-interval negative labels. Predictions remain sealed for full day.

Post-fix check:196.5MiB/1.5GiB, elapsed101s, decoded2463/sampled721, source advanced184.5s. Confirms forward progress and low initial memory; sustained full-day stability still pending.

## 2026-09-09 11:00 follow-up
Same day replay running, OOMfalse. Legacy processed162241 decoded/47281 sampled frames; source1788636079.416 (~03:21), elapsed7198s (~2h). Docker stats801MiB, but cgroup anon~190MiB and filecache~812MiB: resident application memory remains low, no evidence for increasing limit or restarting. Production ROIfalse/oktrue/legacy_cat_crop/5.5FPS. Blind navigation pages72,144,216 reviewed (morningIR/noon/evening): no cat in shown thumbnails, unsampled intervals remain unknown; these are not full-interval negatives. Full-day prediction outputs remain sealed. Next continue existing run and independent reference work; no duplicate jobs.

## 2026-09-09 13:00 follow-up
Same replay running/OOMfalse; legacy decoded329489, sampled96001, elapsed14357s. Source 2026-09-06T06:49:27.072000+08:00 (roughly6h49m source reviewed computationally, NOT visual truth). Production ROIfalse/oktrue/legacy/5.5FPS. Blind navigation pages100-102 reviewed, displayed fountain empty; unsampled intervals unknown. No new accuracy finding, predictions still sealed. No restart or parameter/model change. Next keep run and expand independent review; annotation remains bottleneck, not merely waiting for model output.

## 2026-09-09 15:00 follow-up
Day replay running/OOMfalse, legacy decoded759599/sampled221281, source1788680677.944 (~15:44), elapsed21705s (~6h). No repeat/restart or production changes. Added predeclared scoring/evidence gates to ACCEPTANCE.md before any heldout scoring: visit and bout definitions, one-to-one matching, duplicates/merges, uncertainty, minimum evidence and paired significance. These are engineering gates, not promised model performance; sample-size inadequacy must remain inconclusive. Annotation remains incomplete; do not equate computational coverage with visual ground truth. Next continue existing replay and reference annotation, then freeze candidate config before heldout scoring.

## 2026-09-09 17:00 follow-up
Same container running/OOMfalse. Legacy decoded1022363/sampled297841, elapsed28917s (~8h); source 2026-09-06T21:11:41.232000+08:00. ROI phase has not started yet. Blind pages160-162 reviewed, no cat at fountain in shown frames, not continuous negative truth. No new accuracy finding or production change. Next check transition to ROI and expand independent reference; keep full-day predictions sealed.

## 2026-09-09 19:00 follow-up
Legacy full-day replay finished successfully in34506.7s (~9.59h):1158139 decoded/337400 sampled,1938 timestamp fallbacks, reported gaps0/failures0. Only execution metadata inspected, prediction content remains sealed. Includes previous-day boundary/next-day tail; metric denominators must clip to day. ROI automatically started in same container; source1788625683.408 (~00:28), sampled6721/decoded23065, elapsed1597s; running/OOMfalse. Production ROIfalse/oktrue/legacy/5.4FPS. No model changes or full-day accuracy claim. Next continue ROI and independent annotation. Need inspect timestamp fallback pattern separately before precise onset/duration confidence; no reported failure is not proof every expected source frame decoded.

## 2026-09-09 21:00 independent annotation batch
User explicitly prioritizes real annotation. New raw source20260906000022-60928.mp4 downloaded; all814 sequentially decoded frames visually examined on9 contact sheets (192x108/frame, not sparse temporal sampling). PTS0–60.672s: no cat at fountain,0drinking visits. Labels/hash/frames in reference-sequential/labels.json and all-frames.json. New exhaustive decoded-frame review ~60.7s; prior ~4min night visit remains sampled/dense-boundary review, DO NOT sum as5min exhaustive. Whole-day independent truth still unfinished; remaining almost entire day plus heldout days. ROI job still running, sampled39121/decoded134230, source1788633987.912 (~02:46), elapsed9361s. Next extend sequential source review from00:01:23, and assess practical annotation throughput honestly; do not replace annotations with job status.

## 2026-09-09 23:00 annotation batch
Exact full-resolution pixel hashing of previous814 frames found814 unique frames, so lossless dedup cannot accelerate that clip. Next sequential source20260906000123-60672.mp4 downloaded and all813 decoded frames reviewed on9 sheets. PTS0-60.672: no cat at fountain,0visits. Two sequential clips now have all-decoded-frame review totaling121.344s, no claim about gap between files or previous boundary. Prior4min drinking visit still sampled/boundary review only. New frame map in reference-sequential/20260906000123-60672/all-frames.json, labels appended. ROI running/OOMfalse, sampled69841, decoded239676, source1788641861.856 (~04:57), elapsed16648s. Next source00:02:23/24 per coverage manifest. Annotation pace insufficient for multi-day exhaustive completion; do not claim deadline or full-day improvement.

## 2026-09-10 01:00 follow-up
ROI day replay running/OOMfalse, decoded351722/sampled102481, source1788650227.944 (~07:17), elapsed23842s. First heldout day9/9 now elapsed: launched metadata inventory cat-water-coverage-20260909,0.25CPU/256MB/networknone/source-readonly, outputcoverage-20260909/coverage.json, script audit_coverage_20260909.py. Does not train/tune or inspect heldout predictions. Next check coverage completion and missing/corrupt files. Independent annotation unchanged this batch:121.344s all-decoded-frame review plus prior sampled4min episode; no new accuracy claim.

## 2026-09-10 03:00 follow-up
Heldout9/9 metadata coverage complete exit0:1420files, probeerrors0, covered86303.439972s=99.88824%, no individual gap>2s. This is metadata only, not full decode/visual review. Downloaded coverage-20260909/coverage.json. ROI development replay continues/OOMfalse:515698decoded/150241sampled, source1788662469.456 (~10:41), elapsed30953s. No production changes, labels unchanged. Annotation remains the blocking evidence gap; computational progress must not be described as acceptance. Next prioritize a practical independent-review workflow rather than repeated status-only turns; retain existing jobs and frozen dates.

## 2026-09-10 05:00 scope clarification pending
ROI still running/OOMfalse,709323decoded/206641sampled, source1788676925.472, elapsed38239s. Added REVIEW-STATUS.md with exact annotation limits and no invented metrics. Asked user asynchronously to choose independent day/night sampled assessment vs exhaustive3day review requiring human assistance; no answer yet, no scope change authorized. Existing original acceptance remains in force. Preserve running job/results and await reply for dependent change; do not equate elapsed time with consent. Stop repeating claims that future heartbeat alone solves annotation throughput.

## 2026-09-10 07:00 follow-up
ROI replay continues running/OOMfalse;885663decoded/258001sampled, source1788690089.712 (~18:21), elapsed45479s. Legacy done. No answer to pending scope question, no acceptance change, no new labels or accuracy result. Preserve existing job; do not restart. Next check completion and retain sealed predictions.

## 2026-09-10 09:00 follow-up
ROI existing job still running:996018decoded/290161sampled, source1788698332.8 (~20:38), elapsed52684s. No new labels, no response to pending scope clarification, no changes. Keep predictions sealed and original acceptance requirements. Next check completion; no duplicate job.

## 2026-09-10 11:00 follow-up
ROI running,1101448decoded/320881sampled, source1788706206.696 (~22:50), elapsed59884s. No new labels or scope reply, no accuracy result or production change. Existing run near source end; next inspect completion metadata only, retain predictions sealed.

## 2026-09-10 13:00 full-day replay completed
cat-water-day-replay-20260906 exited0; both legacy and ROI finished. ROI63834.9s (~17.73h), legacy34506.7s (~9.59h), same1158139decoded/337400sampled and1938 timestamp fallbacks, source bounds equal, reported gaps0/failures0. Timing from resource-limited sequential offline runs is not production speed/accuracy proof. Execution summaries local; full-day predictions remain sealed onNAS pending independent reference. Production checked ROIfalse/oktrue/legacy/5.4FPS. No running replay now; DO NOT keep polling/restarting completed job. No scope reply/new annotation. Next dependent work requires chosen review scope or independent reference; next independent collection check is9/10 complete-day metadata after midnight. Completion of this computation is not completion of user validation task.

## 2026-09-10 15:00 follow-up
Production checked ROIfalse/oktrue/legacy/5.4FPS. Completed replay not restarted or polled. Pending scope question has no answer; no new labels, metrics or acceptance claim.9/10 incomplete calendar day; defer complete-day coverage until midnight. State otherwise unchanged.

### 2026-09-10 白天独立复核推进
- 用户要求继续，未改变全天验收范围。导航页168–170仅用于定位，不计连续负样本。
- 独立复核 20260906141258-60928.mp4 全部818解码帧（9页），PTS 0–60.928秒，无猫到饮水机，预测未查看。
- 逐解码帧复核累计3片2445帧、182.272秒；另有夜间抽帧到访参考，不混合计数。全天准确率仍不能得出。
- 未修改生产配置或事件，未重跑已经完成的全天回放。

### 2026-09-10 21:14 independent navigation batch
Production ROI checked enabled=false. Completed replay preserved; no restart. Independently inspected navigation pages120,130,150,180,190,200,210,230,240,260: no cat at fountain in shown frames. Page230 shows lighting/IR transition near source19:31:41, retained as a visibility-transition review target, not negative truth. These sparse pages add zero continuous-reference seconds or definite visits. All-frame reference remains182.272seconds; no accuracy conclusion. Next inspect unreviewed navigation intervals for visits and dense-review the lighting transition; Sep10 full-day coverage only after midnight. No production writes or heldout tuning.

### 2026-09-10 23:14 independent navigation batch
Read acceptance/latest progress. Production ROI verified false. Reviewed previously unreviewed navigation pages50,60,80,90,110,140,220,250,270,280 independently of predictions; no cat at fountain in displayed frames. No continuous negative labels inferred; reference remains182.272seconds and prior sampled night visit. No job started/restarted and no production writes. Sep10 not yet a completed calendar day. Next after midnight inventory Sep10 metadata, and continue dense review of visibility transition identified onpage230. No new accuracy result.

### 2026-09-11 01:14 heldout coverage batch
Read acceptance/latest progress. Production ROI verified false. Sep10 complete: started metadata coverage container cat-water-coverage-20260910 (ID072e51d404a07bbbcf8eac82da78c9acb519cb3ecfc5c9ed0f25c7c8461802be), networknone/0.25CPU/256MB/source recordings RO, outputcoverage-20260910/coverage.json. No duplicate existed; start exit0. Includes Sep9 last clip for boundary. This is metadata only, not reference annotation or model validation. All-frame reference unchanged182.272seconds. Next inspect existing coverage result and continue independent reference, preserve completed sealed development replay. No production changes or heldout tuning.

### 2026-09-11 03:14 Sep10 coverage completed
Existing cat-water-coverage-20260910 exited0:1420files, probeerrors0, covered86302.063972seconds (99.88665%), no individual gap>2seconds, elapsed698.986seconds. Metadata coverage only; cumulative uncovered97.936seconds, not all-frame decode or visual truth. Production ROI checked false. Reference unchanged182.272seconds plus separate sampled night visit. No model/production changes, no replay restart. Next prioritize independent annotation and Sep11 coverage after its day ends; full-day accuracy remains unproven.

### 2026-09-11 05:15 new independent visit candidate
Reviewed navigation pages15,25,35,55 then16–17. Found previously unreviewed cat visit near01:20:33+30s through01:21:34+20s, head down at fountain; absent at next sparse frame +30s. Lapping and precise boundaries remain unknown. Saved pending-visit-0120.json; NOT a confirmed drink and adds zero continuous-reference seconds. Other inspected pages show empty fountain only at sampled instants. Next priority dense independent review of source20260906012033-60928.mp4 and20260906012134-60928.mp4 before any predictions; completed replay remains sealed. No production changes.

### 2026-09-11 07:15 visit source preparation
Downloaded both pending01:20–01:22 source clips (7232799/6381900bytes). Sequentially decoded both and generated9 all-frame contact sheets each; hashes and decode counts saved pending-visit-0120.json. Sheets NOT yet visually reviewed: adds zero reference seconds/confirmed visits. Next inspect these sheets and enlarge mouth-region frames where necessary; do not repeat downloads or count preparation as annotation. Completed jobs preserved, no predictions inspected or production changes.

### 2026-09-11 09:16 independent first-clip review
Visually reviewed all815 decoded frames of20260906012033-60928.mp4 on9 sheets, PTS0–60.928s. Empty initially; cat approaches about23s, close/overexposed occlusion around28.4–29.2s, then head stays down at rim through end. Mouth/licking unresolved at192x108; no confirmed drinking label. Pending reference updated before predictions. Total visual all-frame review243.200s, but latest60.928s includes unresolved behavior and is NOT fully adjudicated truth; prior182.272s negative clips unchanged. Next inspect second clip821frames and enlarge mouth sequence. No production/prediction changes.

### 2026-09-11 11:16 second-clip independent review
All821 decoded frames of20260906012134-60928.mp4 reviewed on9 sheets, PTS0–60.928s. Head down through about22s, lifts22.3–25.6s, departure/occlusion25.9–28.2s, fountain empty28.4s onward. Lapping remains unconfirmed at thumbnail resolution. Both visit clips now visually reviewed but not fully adjudicated drinking truth. Total visual review304.128s, including121.856s unresolved visit clips; established negative clips remain182.272s. Next enlarge mouth-region sequences, not another sparse navigation batch. No predictions inspected or production changes.

### 2026-09-11 13:16 enlarged mouth review
Inspected full-resolution locator and26 consecutive enlarged mouth crops (second clip frames122–147, PTS9.216–10.752). Muzzle at water surface, changing water/spout reflections; cannot reliably separate tongue from water in these frames. Keep visit uncertain; no confirmed drink or added coverage seconds. Saved mouth-sequence-9-11.jpg and crop details. Next inspect another mouth interval nearer head lift before final uncertain classification. No predictions/production changes; completed jobs not restarted.

### 2026-09-11 15:16 visit0120 independently frozen uncertain
Inspected26 additional mouth crops frames285–310, PTS21.248–22.784, second clip. Head visibly lifts near22.016s, but tongue not unequivocally separable from water. Final independent classification: confirmed fountain visit, drinking uncertain. Frozen reference-visit-0120-frozen.json +sha BEFORE predictions. Keep in uncertainty bounds, not definite TP/FN or automatic negative. No added temporal coverage;304.128s visually reviewed still includes unresolved behavior. Next move to other independent visits; do not keep revisiting these crops or claim full-day improvement. No production changes.

### 2026-09-11 17:17 new daytime visit candidate
Production ROI verified false. Independently inspected navigation pages65,75,85,95. Page95 reveals daylight visit around08:03: cat approaches, muzzle at water, apparent tongue in+50s frame, then leaves. Saved pending-visit-0803.json with exact source files. Not yet definite drinking label or continuous truth; adds zero reviewed seconds. Next prioritize raw dense review of this daytime visit; do not revisit frozen uncertain01:20visit. Other displayed navigation frames empty only at sampled instants. No predictions inspected or production changes.

### 2026-09-11 19:17 daytime raw sequence check
Downloaded middle clip20260906080330-60928.mp4 (11977057bytes; hash saved pending-visit-0803.json). Sequentially decoded/viewed26 frames654–679, PTS49.152–50.944: head already lifted, unlike approximate navigation+50s image. Navigation offsets are approximate keyframe indexing and must not define truth boundaries. No confirmed drinking label from this probe. Next all-frame/dense review earlier middle-clip section to locate actual lapping. No production/prediction changes.

### 2026-09-11 21:18 daytime lapping supported
Decoded middle08:03:30clip814frames/9sheets. Independently viewed first400frames (pages0–3, PTS0–29.952): repeated pink tongue extension toward water supports sustained lapping. Saved supported interval, onset beforeclip/end stillpending. Remainingpages4–8 NOT reviewed this batch; do not countwholeclip. Prior49.152–50.944probe remains separate and overlaps futurefullreview. Next finishpages4–8 then neighboringclips to definevisitbounds beforepredictions. No model/productionchanges; evidence stilldevelopmentonly, notfull-dayacceptance.

### 2026-09-11 23:20 daytime middle clip fully reviewed
Viewed remainingpages4–8, frames400–813. All814frames/60.928s of08:03:30middleclip now reviewed, including previously viewed49s probe (do not doublecount). Lapping supported through44.8s; head lift boundary44.8–46.336, raised46.336–52.48, resume boundary52.736–53.504, renewed lapping53.504–60.928. Bout boundaries approximate, same visit; onset/offset still require neighboringclips. Updated independent draft, predictions unseen. Total six fully viewed source clips365.056s includes uncertain visit, not all adjudicated drinking truth. Next neighboring08:02:29/08:04:31clips and dense boundary checks; Sep11 coverage after midnight. No productionchanges.

### 2026-09-12 01:21 heldout Sep11 coverage started
Reconnected SSH after expired session; synced pending daytime independent reference from prior batch. Production ROI verified false. Sep11 calendar day finished: no existing duplicate job; launched cat-water-coverage-20260911, ID1c5b43229fe4ffc5f36161eb2e6a266aa8bc2a79e892806cc4f0065d02049c97, start exit0, networknone/0.25CPU/256MB/sourceRO. Metadata inventory only, not accuracy or independent annotation. Visual coverage remains365.056s across six fully viewed clips including uncertain behavior; no new visual labels this batch. Next fetch completed Sep11 coverage and prioritize neighboring08:02:29/08:04:31clips for pending daytime visit bounds. No production writes or completed replay restarts.

### 2026-09-12 03:21 Sep11 coverage complete and daytime onset review
cat-water-coverage-20260911 exited0:1419files, probeerrors0, coverage86303.271972s (99.88805%), no individualgap>2s, elapsed901.243s. Saved local coverage.json; metadata only. Downloaded preceding08:02:29clip8731122bytes and decoded818frames/9sheets. Independently viewed pages0–1, frames0–199, PTS0–14.848: no cat at fountain. Remainingpages2–8 unreviewed; onset remains pending, no additional full clip counted (six fully viewed clips365.056s unchanged). Reference saved before predictions. Next finish precedingclip and fetch08:04:31endclip. No production changes or replay restarts.

### 2026-09-12 05:22 daytime preceding clip fully reviewed
Viewed remainingpages2–8 of08:02:29source, all818frames now reviewed. Empty fountain through37.632s, approach about38–50s, muzzle near rim50.688–52.48 without definite lapping, close-lens reposition53–55s, muzzle lowers58.624–59.648, visible pink tongue supports lapping60.160–60.928. Onset uncertainty58.624–60.160 retained; earlier rim contact needs enlargement, not automatic drinking. Seven fully viewed clips total425.984s, includes uncertain behavior; no full-day accuracy claim. Next fetch/review08:04:31clip and enlarge onset boundaries before freezingvisit. Predictions unseen, no production changes.

### 2026-09-12 07:23 daytime ending clip fully reviewed
Downloaded08:04:31source8433743bytes, decoded821frames and independently reviewed all9sheets. Repeated visible tongue0–2.048s; stops/lifts2.304–2.560 (uncertain end2.048–2.560), upright then departs8–14.848, empty15.104–60.928 with no return. All three daytimevisit clips now fully viewed, draft saved before predictions; boundary enlargement remains before freeze. Eight fully viewed sourceclips total486.912s including uncertain behavior, not all adjudicated drinking truth. Next enlarge preceding50–52/58–60 and middlepause/endboundaries, then freeze reference. No production changes, no jobs restarted.

### 2026-09-12 09:23 enlarged early approach check
Viewed31 consecutive enlarged crops of precedingclip frames674–704, PTS50.176–52.224. Muzzle remains above/outside water while approaching and moving along rim; no tongue-water contact. Supports non-drinking approach, no earlier lapping bout added. Saved day0803-onset-50-52.jpg and reference observation before predictions. Reviewed temporal coverage unchanged486.912s. Next enlarge58–60onset, middlepause and final2s boundary, then freeze daytimevisit. No production changes or new jobs.

### 2026-09-12 11:23 daytime onset refined independently
Viewed26 consecutive enlarged crops frames783–808 of precedingclip. First visible pink tongue contact frame799 atPTS59.648, repeated804/808; narrowed onset bracket to59.392–59.648 (PTS quantized). Supportedlapping now59.648–60.928, replacing thumbnail60.160start. Saved evidence and reference before predictions. No added temporal coverage;486.912s reviewed. Next middlepause/endboundary enlargement then freeze daytimevisit. No productionchanges or newjobs.

### 2026-09-12T11:32 daytime independent reference frozen
Responding to user: completed65 enlarged boundaryframes (pause29/resume20/end16) without predictions. One definite daytimevisit, two lappingbouts with pause; onset59.392–59.648 precedingclip, pause44.8–45.312 middleclip, resume52.992–53.248, finalend2.048–2.560 endingclip. Frozen reference-visit-0803-frozen.json +SHA. Eight fully viewed clips486.912s unchanged (boundary crops overlap), not full-day truth. Next new independent review batch and separately score frozen developmentvisit. Production unchanged.

### 2026-09-12T11:38 heldout continuous annotation batch001
User explicitly requests full-day annotation. Switched priority to heldoutSep9 chronological review; do not return to frozen development08:03example. Reviewed31contact sheets: prior-dayclip frames500–809 and all816/812/819frames of first3Sep9clips. All show no cat at fountain. Saved heldout-reference-20260909/labels.json with raw hashes/frame ranges/PTS and exact merged reviewedseconds 199.27199983596802. Source spans through00:03:19.928; tiny gaps explicitly remain outside reviewed intervals. queue.json tracks all1420sources, unreviewed remain unknown. Next start20260909000320-60928.mp4 and continue chronological batches; no predictions seen or heldout tuning. No full-day truth yet.

Heldout batch002 preparation launched: cat-water-heldout-prep-002 IDc33eb6b9381737f9707c48bf6d869740a73475b74c5f400074756a07c765ae5a, running confirmed, sourceRO/networknone/0.5CPU/512MB. Prepares next30chronological sources (coverage rows4:34), NO automatic labels. On next run fetch prepared sheets starting20260909000320-60928 and visually review entire consecutive batches, then update labels/queue. Completed Sep9–11 coverage/replay need no routine rechecking. ProductionROI verifiedfalse.


### 2026-09-12 独立标注续接
已查看并录入 Sep9 00:03:20、00:04:21、00:05:22 三段全部连续解码帧，均为饮水机前无猫。heldout 累计381.800秒，下一段00:06:23；未查看区域与录像缺口保持未知。批次002共30段联系表已准备并取回，其余未查看片段不能计作标注。全天标注尚未完成，不能给出全天准确率。

### 2026-09-12 15:25 heldout batch visual review
Independently viewed all18 contact sheets of00:06:23 and00:07:24, all815+819 consecutive frames; fountain empty throughout. Saved hashes and PTS, reviewed total 503.40000000000003 seconds. Next 20260909000825-60928.mp4. No predictions viewed, no heldout tuning, no production changes. Full-day annotation incomplete.

### 2026-09-12 17:26 heldout chronological review
Independently viewed all18 contact sheets of00:08:25 and00:09:26, 1626 consecutive decoded frames. Fountain empty throughout; background person movement in first clip does not approach fountain. Saved hashes, actual frame ranges and PTS; reviewed total 624.744 seconds, source through00:10:26.672. Next 20260909001027-60928.mp4. Gaps and remainder unknown. No predictions viewed, heldout tuning or production changes; full-day truth incomplete.

### 2026-09-12 19:26 heldout chronological review
Independently viewed all18 contact sheets of00:10:27 and00:11:28, 1630 consecutive decoded frames. No cat at fountain throughout both clips. Saved hashes, frame ranges and PTS; reviewed total 746.6 seconds. Next 20260909001228-60928.mp4. Gaps and unreviewed remainder unknown; no predictions, tuning or production changes. Full-day annotation incomplete.


独立连续复核更新：00:12:28 和 00:13:29 两段全部 18 张连续帧联系表已目视查看，1631 帧均未见猫在饮水器处。按实际 PTS 合并重叠区间，累计 867.2720000000002 秒；下一段 20260909001430-60928.mp4。未查看模型预测，全天标注仍未完成。


### 2026-09-12 23:28 独立连续复核
00:14:30 和 00:15:31 两段全部18张连续帧联系表已目视查看，815+817帧均未见猫在饮水器处。按实际PTS合并区间，累计 989.1280000000002 秒；下一段 20260909001632-60928.mp4。未查看模型预测，未调整模型；缺口和余下录像保持未知，全天标注未完成。


### 2026-09-13 01:29 独立连续复核
00:16:32、00:17:33全部18张连续帧联系表已目视查看，[820, 809]帧均无猫在饮水器处。累计复核 1110.4720000000002 秒，下一段 20260909001834-60672.mp4。哈希与实际PTS已记录，缺口/余下未知；未读预测、未调参，全天未完成。


### 2026-09-13 03:30 独立连续复核
00:18:34、00:19:35全部18张连续帧联系表已目视查看，[817, 811]帧均无猫在饮水器处。累计复核 1232.0720000000003 秒，下一段 20260909002036-60928.mp4。哈希与实际PTS已记录，缺口/余下未知；未读预测、未调参，全天未完成。

### 独立标注续接：00:20:36—00:22:37
已逐页查看两个完整片段的全部连续帧（18 页），饮水器区域均无猫；未查看模型预测。标注累计有效覆盖 1353.928 秒，仍未完成全天；下一片段 20260909002237-60928.mp4。

独立复核 20260909002237-60928 全部 817 帧：猫从饮水器后方经过，未停下舔水；记录为经过但未饮水。累计 1413.672 秒（扣除片段重叠）；下一片段 20260909002338-60928.mp4。

### 2026-09-13 07:32 独立连续复核
00:23:38、00:24:39全部18张连续帧联系表已目视查看，[814, 818]帧均无猫在饮水器处。累计复核 1535.528 秒，下一段 20260909002540-60928.mp4。哈希与实际PTS已记录，缺口及余下未知；未读预测、未调参，全天未完成。

### 2026-09-13 09:33 独立连续复核
00:25:40、00:26:41全部18张连续帧联系表已目视查看，[818, 811]帧均无猫在饮水器处。累计复核 1656.872 秒，下一段 20260909002742-60672.mp4。哈希与实际PTS已记录，缺口及余下未知；未读预测、未调参，全天未完成。

### 2026-09-13 11:33 独立连续复核
00:27:42、00:28:43全部18张连续帧联系表已目视查看，[815, 813]帧均无猫在饮水器处。累计复核 1778.472 秒，下一段 20260909002944-60928.mp4。哈希与实际PTS已记录，缺口及余下未知；未读预测、未调参，全天未完成。

### 2026-09-13 15:50 独立连续复核
00:29:44、00:30:45 全部18张连续帧联系表已目视查看，822及811帧饮水器处均无猫。累计有效复核 1900.328 秒，下一段 20260909003146-60928.mp4。哈希与实际PTS已记录；未读预测，全天未完成。

### 2026-09-13 17:52 独立连续复核
00:31:46、00:32:46 全部18张联系表已目视查看，818及813帧饮水器处均无猫。累计有效复核 2021.256 秒（重叠扣除），下一段 20260909003347-60928.mp4；准备批次002已经全部复核，后续需准备新批次。未读预测、未调参，全天未完成。

下一批准备已启动：cat-water-heldout-prep-003，容器ID 37cbe64d7c34515ae9a3f21bd5bc9d756f09c226f0b1917d31f80a89c5e5f63f；coverage rows[34:64]，断网、0.5 CPU、512MB，录像只读。prepare_heldout_batch003.py 仅生成全帧联系表，未标注；勿重复启动。完成后下载这30个目录再继续视觉复核。

### 2026-09-13 20:00 独立连续复核
批次003已退出0，30段联系表已下载解压到本地（heldout-prep-003.tar.gz，285076962字节），勿重跑。实际目视完成00:33:47、00:34:48 全部18页，819及816帧，饮水器处均无猫。累计有效覆盖 2142.856 秒；下一段 20260909003549-60928.mp4。其余28段只是已准备，不是标注。未查看预测，全天未完成。

### 2026-09-13 21:54 独立连续复核
00:35:49、00:36:50 全部18张联系表已目视查看，两个片段各813帧饮水器处均无猫。累计有效覆盖 2264.200 秒；下一段 20260909003751-60928.mp4。未查看预测，未调参，全天未完成。

### 2026-09-13 23:54 独立连续复核
00:37:51、00:38:52 全部18页已目视查看，815及820帧饮水器处均无猫。累计有效覆盖 2386.056 秒；下一段 20260909003953-60928.mp4。未查看预测、未调参，全天未完成。


## 2026-09-13T23:58:45.414941+08:00 独立标注续审
实际逐页查看 00:39:53、00:40:54、00:41:55、00:42:55、00:43:56 五段共 45 张连续帧图，均未见猫到饮水器，已保存独立标签。累计有效已审 2689.7680000000005 秒；下一段 20260909004457-60928.mp4。全天尚未完成，不支持准确率提升结论。


### 2026-09-14T01:54:36.153821+08:00 独立连续复核
00:44:57、00:45:58 全部18页已目视查看，814及811帧均无猫在饮水器处。累计有效已审 2811.1120000000005 秒；下一段 20260909004659-60672.mp4。未读预测、未调参；全天标注尚未完成。


Independent annotation checkpoint: saved 20260909004659 and 20260909004800, all 18 contact sheets reviewed. Cat passes behind fountain, no drinking observed. Reviewed interval union 2932.712 seconds; next 20260909004901-60928.mp4. Partial only; no full-day accuracy claim.


### 2026-09-14 05:55 独立连续复核
00:49:01、00:50:02 全部18页实际目视，814及818帧均无猫在饮水器处。累计有效已审 3054.568 秒；下一段 20260909005103-60928.mp4。未读预测、未调参；全天尚未完成。


### 2026-09-14 07:56 独立连续复核
00:51:03、00:52:04 全部18页实际目视，815及821帧。猫从左向右走过后坐在饮水器后右方，头部抬高远离水面，未见饮水；人从背景经过。累计有效已审 3176.424 秒；下一段 20260909005305-60928.mp4。未读预测、未调参；全天尚未完成。


### 2026-09-14 09:57 独立连续复核
00:53:05、00:54:05 全部18页实际目视。猫起身向右离开，未见饮水，之后饮水器处无猫。累计有效已审 3297.096 秒（交叠已去重）；下一段 20260909005506-60928.mp4。未读预测、未调参；全天尚未完成。


### 2026-09-14 11:59 独立连续复核
00:55:06、00:56:07 全部18页实际目视，810及816帧，饮水器处均无猫。累计有效已审 3418.440 秒；下一段 20260909005708-60928.mp4。未读预测、未调参；全天尚未完成。

2026-09-14：独立逐帧查看并保存 00:57:08、00:58:09 两段共 1632 帧，饮水机处均无猫；未查看预测。9月9日已审有效时长 3540.296 秒，下一段 20260909005910-60928.mp4。全天标注与准确率验证仍未完成。

2026-09-15 00:00：00:59:10 全部9页816帧实际目视，饮水机处无猫。未读预测。有效已审 3601.224 秒；下一段 20260909010011-60928.mp4，全天未完成。

2026-09-15：01:00:11 全部9页817帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 3662.152 秒；下一段 20260909010112-60928.mp4，全天未完成。

2026-09-15：01:01:12 全部9页822帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 3723.080 秒；下一段 20260909010213-60928.mp4，全天未完成。

2026-09-15：01:02:13 全部9页812帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 3784.008 秒；下一段 20260909010314-60928.mp4，全天未完成。

2026-09-15：01:03:14 全部9页817帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 3844.680 秒；下一段 20260909010414-60928.mp4，全天未完成。

2026-09-15：批次004联系图准备已启动（coverage rows[64:94]，容器 9ab86d162d67）。录像只读、断网、0.5 CPU、512MB；该作业只生成逐帧联系图，不生成标签，也不影响生产服务。

2026-09-15：01:04:14 全部9页808帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 3904.680 秒；下一段 20260909010515-60672.mp4，全天未完成。

2026-09-15：01:05:15 全部9页817帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 3965.352 秒；下一段 20260909010616-60928.mp4，全天未完成。

2026-09-15：01:06:16 全部9页814帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 4026.280 秒；下一段 20260909010717-60928.mp4，全天未完成。

2026-09-15：01:07:17 全部9页818帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 4087.208 秒；下一段 20260909010818-60928.mp4，全天未完成。

2026-09-16：01:08:18 全部9页814帧实际目视。猫约在 PTS 28–45 秒占据饮水器位置，但口鼻与水面被饮水器边缘全程遮挡，不能确认舔水；按“不可判断”记录，未计入饮水或阴性样本。未读预测、未调参。有效已审 4148.136 秒；下一段 20260909010919-60928.mp4，全天未完成。

2026-09-16：01:09:19 全部9页816帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 4208.808 秒；下一段 20260909011020-60928.mp4，全天未完成。

2026-09-16：01:10:20 全部9页820帧实际目视。猫约 PTS 29 秒起至片段末持续可见舔水，记为一段明确饮水且跨片段未结束；未读预测、未调参。有效已审 4269.736 秒；下一段 20260909011121-60928.mp4，全天未完成。

2026-09-16：01:11:21 全部9页813帧实际目视。猫从片段首帧至末帧均持续可见舔水，是上一段明确饮水的延续且跨片段未结束；未读预测、未调参。有效已审 4330.664 秒；下一段 20260909011222-60928.mp4，全天未完成。

2026-09-17：01:12:22 全部9页816帧实际目视。猫从片段首帧持续可见舔水至约 PTS 11.776 秒，约 PTS 12.032 秒转身离开，后续画面饮水机处无猫；与前两段合并为同一次明确饮水。未读预测、未调参。有效已审 4391.336 秒；下一段 20260909011323-60928.mp4，全天未完成。

2026-09-17：01:13:23 全部9页809帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 4452.008 秒；下一段 20260909011423-60672.mp4，全天未完成。

2026-09-17：01:14:23 全部9页819帧实际目视，饮水机处无猫；未读预测、未调参。有效已审 4512.680 秒；下一段 20260909011524-60928.mp4，全天未完成。

2026-09-18：用户明确要求推进识别提升，恢复开发诊断。已核实线上 YOLOv8n、ROI disabled；从 recent-clips.tar 恢复7段开发正例，按原标签区间截取6段负例。隔离容器重新推理和时序模拟：阈值0.90触发3/7正例、0/6负例；0.80触发6/7、0/6。样本来自已有事件且短负例，属于开发结果，不是全天验收。固定0.80为离线候选，启动9月6日完整实时流程回放，容器 cat-roi080-development-20260918，输出 accuracy-20260908/recovery-20260918/development-day-roi080.json，进度同名.progress.json。录像只读、断网、1 CPU、1536MB；正式候选仍关闭。下次先检查该作业是否确实推进/完成/失败并处理，不重复启动、不只回复状态；随后继续真实独立标注。

2026-09-18：01:15:24 全9页812帧实际目视，无猫在饮水机处。未读预测。已审时长改为源时间区间并集，去除片段重叠重复计时，当前 4572.936 秒；下一段 20260909011625-60928.mp4。

2026-09-18：01:16:25 全9页819帧实际目视，饮水机处无猫；未读验收预测。有效已审4633.864秒，下一段 20260909011726-60928.mp4。开发样例对照容器 cat-reference-0803-20260918 已启动，本次读取结果，不重复启动。

2026-09-18 开发原始连续录像对照完成：reference-0803 legacy/ROI均4082解码帧、1191采样帧，无解码失败。已冻结参考1次饮水、支持舔水55.808秒；两流程均命中1次，无重复事件。legacy有效舔水1.0秒（绝对误差54.808秒，98.21%），ROI0.80为58.5秒（绝对误差2.692秒，4.82%）。该单一开发样例证明时长记录改善，不证明全天召回提高，不作验收结论。完整开发日回放进度source_time1788641923.368（2026-09-06 04:58:43+08:00），仍未完成。下一步继续独立标注与开发日误差分析。

2026-09-18：01:17:26、01:18:27共18页实际目视，813及817帧，饮水器处均无猫；未读验收预测、未调参。有效已审4755.464秒；下一段20260909011928-60928.mp4。

2026-09-18：01:19:28、01:20:29共18页实际目视，帧数[818, 815]，饮水器处均无猫；未读验收预测、未调参。有效已审4877.32秒；下一段20260909012130-60928.mp4。

2026-09-18 用户要求继续算法优化：新增持续证据门离线原型 sustained_gate.py（2秒窗口、80%时间支持、缺帧清空、时间加权）。开发扫描0.625门槛7/7正片段出现证据、0/6负片段；尚未接入完整实时状态机，不能作为事件召回提升结论。4项时序边界测试通过。详见 recovery-20260918/SUSTAINED-EXPERIMENT.md。生产未修改，原ROI0.80全天回放仍运行无最终文件。

2026-09-18：01:21:30、01:22:31共18页实际目视，帧数[817, 811]，饮水器处均无猫；未读验收预测、未调参。有效已审4998.664秒；下一段20260909012332-60672.mp4。


### 2026-09-19 持续证据算法接入离线完整流程
已从 NAS 当前 detection_worker.py 创建隔离副本，ROI 原高分门控保留，另加阈值 0.625、2 秒内 80% 连续证据分支；经原事件候选/持续时间状态机输出。时间间断及重置清空积累，活动饮水时长仍按原高分证据计量，需单独验证。生产未修改。开发 0803 参考段回放已启动：cat-sustained-reference-20260919，输出 recovery-20260918/reference-0803-sustained.json。尚无最终结果，不能声称召回提升。原 ROI0.80 全天回放仍在运行。

2026-09-19：01:23:32与01:24:32两段全18页完成目视复核，共1630帧，均无猫在饮水器处；未读验收预测。实际末PTS分别60.672/60.928秒。有效已审5119.592秒；下一段20260909012533-60928.mp4。

2026-09-19：01:25:33、01:26:34共18页实际目视，帧数[820, 811]，饮水器处均无猫；未读验收预测、未调参。有效已审5241.448秒；下一段20260909012735-60928.mp4。

2026-09-19：01:27:35、01:28:36共18页实际目视，帧数[818, 816]，饮水器处均无猫；未读验收预测、未调参。有效已审5363.048秒；下一段20260909012937-60928.mp4。

2026-09-19：01:29:37、01:30:38共18页实际目视，帧数[817, 815]，饮水器处均无猫；未读验收预测、未调参。有效已审5484.648秒；下一段20260909013139-60928.mp4。

2026-09-19：01:31:39、01:32:40共18页完成实际目视复核，帧数[813, 818]，饮水器处均无猫；未读验收预测、未调参。有效已审5611.616秒；下一段20260909013341-60928.mp4。

2026-09-22：01:33:41 全9页811帧实际目视，无猫在饮水器处；未读预测。有效已审5672.544秒，下一段20260909013441-60928.mp4。此前已下载但未实际复核的后续录像仍保持未标注。

2026-09-22：下一段01:34:41联系图不存在（HTTP404），不能标注。已确认004准备作业正常结束，启动005：coverage rows[94:124]，容器cat-water-heldout-prep-005，ID e7b6606a0769a36722645f53794991187396f175918621a3d9ec5780a9d261e6；0.5 CPU、512MB、断网、录像只读。仅准备图片，不增加已审覆盖。下次从01:34:41实际目视继续，勿重复启动005。

2026-09-22：01:34:41全9页819帧实际目视，无猫在饮水器处；未读预测。图片已补齐，无新增准备作业。有效已审5732.544秒，下一段20260909013542-60928.mp4。

2026-09-22 用户要求以准确率提升为主：取回完整ROI0.80开发日回放，1158139解码帧/337400采样帧，与旧版相同，无解码失败。已发现持续证据门与候选持续时间衔接导致已知21:45:50开发正例仍漏检。新增离线副本sustained_hold_detection_worker.py：仅在2秒80%支持成立后保持0.8秒，缺帧/时间回退/重置清空。开发分数轨迹经原时序门与候选门对比，正例6/7→7/7，6段负例仍0触发；这是门控级开发回归，非完整流程/全天证明。生产未变。下一步必须对该漏检原始连续录像跑完整流程，并评估全天误触发，不再仅增加空画面标注。

2026-09-22 完整main流程开发验证完成：0.625+0.8秒保持仍漏目标；开发阈值调0.55后目标检出1次，其余6正例均检出1次，6负例均0事件，无解码失败。结果gate-validation-combined055-results.json，报告VALIDATION-20260922.md。负例仅25.2秒且重编码短片，不能证明全天误报。目标active_drink_sec仍0，计时待修。生产ROI仍关闭。下一步原始连续录像及更长独立负例验证，不以门控扫描替代main流程。

2026-09-23：01:35:42全9页812帧实际目视，无猫在饮水器处；未读预测。有效已审5793.472秒，下一段20260909013643-60928.mp4。

2026-09-23：01:36:43全9页819帧实际目视，无猫在饮水器处；未读预测。有效已审5854.144秒，下一段20260909013744-60928.mp4。全天验收未完成。

2026-09-23：01:37:44全9页815帧实际目视，无猫在饮水器处；未读预测。末PTS60.928秒，有效已审5915.072秒，下一段20260909013845-60928.mp4。全天验收未完成。
