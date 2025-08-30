# CHANGELOG

## v1.16.4 (2025-08-30)

### Fix

* fix: change poetry run to pip install ([`40bc0c0`](https://github.com/agbleze/objectscope/commit/40bc0c01fb1dcd4500d836da209ac51d6c387097))

## v1.16.3 (2025-08-30)

### Fix

* fix: remove poetry shell ([`98251d0`](https://github.com/agbleze/objectscope/commit/98251d0db7dede2502bf9a2e88b40b2b2fd67c2f))

## v1.16.2 (2025-08-30)

### Fix

* fix: use poetry to manage pip install ([`5e2a6d1`](https://github.com/agbleze/objectscope/commit/5e2a6d1e6d65220b221552425c511918b244be9d))

## v1.16.1 (2025-08-30)

### Fix

* fix: update ci-cd.yml ([`f4a3ed9`](https://github.com/agbleze/objectscope/commit/f4a3ed9d03e601b18353406390262ef1cb6b650c))

* fix: remove unused result from create_trainer ([`068702b`](https://github.com/agbleze/objectscope/commit/068702b81e2e814487343bc00d59f0d8951e812f))

* fix: remove repo after publishing to testpypi as fix for poetry install ([`3ba794b`](https://github.com/agbleze/objectscope/commit/3ba794b15b49d528b6a16a66f112394c85f26de4))

### Style

* style: remove comment ([`37910c2`](https://github.com/agbleze/objectscope/commit/37910c2db941e5a57745c75da79f19c0bed13da2))

## v1.16.0 (2025-08-30)

### Feature

* feat: add save_class_metadata ([`20e0e18`](https://github.com/agbleze/objectscope/commit/20e0e188cabc84f6f662a5ea744038a4a5ffd7c7))

* feat: add save_class_metadata_as ([`56668fd`](https://github.com/agbleze/objectscope/commit/56668fdda856ff349900fa0ae080c65d5a78141f))

### Fix

* fix: update creating class_metadata_map and dumping it ([`4f62afb`](https://github.com/agbleze/objectscope/commit/4f62afb05ccc1243bb72580a3999da488e7f2d66))

### Unknown

* chores: add tests for save_class_metadata ([`8dc54c1`](https://github.com/agbleze/objectscope/commit/8dc54c19cf46ef612341761409632878de280426))

* chores: add tests for anchor miner ([`c16968a`](https://github.com/agbleze/objectscope/commit/c16968ad39f80b1c8ac6d15695cb8b6bd81952df))

## v1.15.0 (2025-08-30)

### Feature

* feat: add AnchorMiner ([`3658046`](https://github.com/agbleze/objectscope/commit/3658046f1d452f320dfd1066026e1b774cdf4915))

* feat: add get_size_ratio_fitness_score ([`8dfa6c0`](https://github.com/agbleze/objectscope/commit/8dfa6c01b84e5fa4d8e1c660270d3215a13c276a))

## v1.14.1 (2025-08-29)

### Fix

* fix: add gt_sizes and ratios ([`7b4bd91`](https://github.com/agbleze/objectscope/commit/7b4bd91a4d18d5c4244dd1b1a17f6fe28d69e714))

### Style

* style: remove commented code ([`8de17d9`](https://github.com/agbleze/objectscope/commit/8de17d93b05b0ba5cbc91673dde6ed01fb00f2e3))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main

chores: update with remote ([`8634deb`](https://github.com/agbleze/objectscope/commit/8634deba20fef670ffb2b478b37acec15b553c3a))

## v1.14.0 (2025-08-26)

### Feature

* feat: add coco_annotation_to_df ([`a4abc14`](https://github.com/agbleze/objectscope/commit/a4abc14f0e8ba5c955662f6b025ef083b27c2e67))

### Unknown

* chores: add dockerfile for gpu torch tensorflow ([`65d7c5b`](https://github.com/agbleze/objectscope/commit/65d7c5b97348bf667c4007b87ef9427899e4207b))

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
chores: update with new release ([`5990db8`](https://github.com/agbleze/objectscope/commit/5990db87e8f1b9183d20310384819a2c8ee3ff71))

## v1.13.0 (2025-08-23)

### Feature

* feat: add anchor_bbox_utils ([`b6012b5`](https://github.com/agbleze/objectscope/commit/b6012b51d67d135f044b20e312a30ce9932b3738))

* feat: add anchor sizes and ratios ([`ead5259`](https://github.com/agbleze/objectscope/commit/ead5259ce6bdc550e05836798f3bca0c1cbc4bb7))

* feat: add draw_bbox_and_polygons ([`e190329`](https://github.com/agbleze/objectscope/commit/e19032954d8ea7a2b36447b8b0929abf33c7769e))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
chores: reconcile with remote ([`22869e4`](https://github.com/agbleze/objectscope/commit/22869e433dbfdc888168e176b4d772e859a208aa))

## v1.12.0 (2025-08-22)

### Feature

* feat: add predict_bbox ([`e17b884`](https://github.com/agbleze/objectscope/commit/e17b8842e4b9f2cb163a6880412168ac191e56b5))

* feat: add dynamic axis to export_to_onnx ([`9a40bcc`](https://github.com/agbleze/objectscope/commit/9a40bcc9f450ab42797a76d35753c51a0f9a79fa))

## v1.11.1 (2025-08-21)

### Fix

* fix: update OnnxModelExporter has no inputs attribute ([`2b498b8`](https://github.com/agbleze/objectscope/commit/2b498b8974596dd22712c14bcadcafd0078b55bd))

## v1.11.0 (2025-08-20)

### Feature

* feat: add logging to create_trainer ([`e5f3229`](https://github.com/agbleze/objectscope/commit/e5f3229af706ca6946ce78e7c832e8170c8c3a9a))

* feat: add logging to get_best_model ([`e2302a8`](https://github.com/agbleze/objectscope/commit/e2302a82395a28885cc26ccbf6f8d31909e10ae5))

### Fix

* fix: change tensorboard launch commard ([`91eae44`](https://github.com/agbleze/objectscope/commit/91eae44b0a2c8172506fccbbc918c8d41adf5313))

* fix: add reuse of existing of cfg tp create_trainer ([`edc8775`](https://github.com/agbleze/objectscope/commit/edc87753c9caa6d050c3061593db8fc69dce0824))

## v1.10.0 (2025-08-19)

### Feature

* feat: add compute_statistics ([`ecea3d9`](https://github.com/agbleze/objectscope/commit/ecea3d9ab6c07cbc584af4be1371b7f5e01c7d1f))

## v1.9.0 (2025-08-18)

### Build

* build: change pip to poetry for install from testpypi ([`ce19a93`](https://github.com/agbleze/objectscope/commit/ce19a93e0f9f73b8a28bb566760527e8403b22eb))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
chores: reconcil remote 1.8 ([`970acb4`](https://github.com/agbleze/objectscope/commit/970acb46872757a86e497455c6e4f680f9211180))

## v1.8.0 (2025-08-18)

### Feature

* feat: add save_model ([`8514da9`](https://github.com/agbleze/objectscope/commit/8514da930375a44ed0be88234ea1e40c501c4c2f))

* feat: add create_script_model ([`8193922`](https://github.com/agbleze/objectscope/commit/81939224272c7636fb8135b5224dee28a2878528))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
chores: update with remote ([`a354752`](https://github.com/agbleze/objectscope/commit/a354752e61fba36b41b87e17ee6f8c1abacd1185))

## v1.7.0 (2025-08-17)

### Build

* build: update to use single run for CI ([`382e5a4`](https://github.com/agbleze/objectscope/commit/382e5a475d89010b98b77ced2ee7c7dad6e7ead3))

* build: update with collapsible section to rn commands ([`0dc0ad3`](https://github.com/agbleze/objectscope/commit/0dc0ad3ae58ec3e9f47d5144156adbe0b7586ee5))

* build: updtae with install poetry with pip ([`d0d397c`](https://github.com/agbleze/objectscope/commit/d0d397c4e7585f5e5f00af8cc9ef9099d6484b13))

* build: update with create venv and move to GITHUB_PATH ([`27e5786`](https://github.com/agbleze/objectscope/commit/27e5786ee8bb8416a8567d4e2df15cea4185ce69))

* build: add sphinx to docs requirement ([`d5d4d02`](https://github.com/agbleze/objectscope/commit/d5d4d0293e7bb287258cfa05dbd55c34f0e69290))

* build: change poetry virtualenvs to false ([`9936402`](https://github.com/agbleze/objectscope/commit/9936402110177e28b2186783b432a7028430f309))

* build: move poetry run test to installation stage ([`a9c828f`](https://github.com/agbleze/objectscope/commit/a9c828f06a0d66e2ab8f21c40b017c349527c36c))

* build: update poetry use env ([`2ebd560`](https://github.com/agbleze/objectscope/commit/2ebd560187bfa441948cf375a4262b7ec980065c))

* build: update poetry use env ([`c89a0cd`](https://github.com/agbleze/objectscope/commit/c89a0cdf0bc20d8f66ab2c66a935df98c4337b3b))

* build: update test command ([`eb8e663`](https://github.com/agbleze/objectscope/commit/eb8e6637a244c57a04ae4234705705038083a9a2))

* build: create venv in cicd ([`8f8b129`](https://github.com/agbleze/objectscope/commit/8f8b129f76f386d94356e488281b3ae40a4b23cf))

* build: update dependencies ([`5a05d78`](https://github.com/agbleze/objectscope/commit/5a05d789c984e589657a61afa0194b1f6bf4f22e))

* build: update lock ([`88d6de9`](https://github.com/agbleze/objectscope/commit/88d6de9a253ad4cabf6ba5132a6c3db3852b0d41))

* build: update dependencies ([`77b6a4e`](https://github.com/agbleze/objectscope/commit/77b6a4ef0e61e39ed8cd58d9f12b6635259c2cc0))

* build: install wheel ([`28f1e9d`](https://github.com/agbleze/objectscope/commit/28f1e9dd443279e4d3e4099a74fedbd9d910cb36))

* build: update with no-build-isolation ([`0e7e80e`](https://github.com/agbleze/objectscope/commit/0e7e80e5de0206bb80636e4cfab3787afc2afecb))

* build: update cicd ([`a4b78e3`](https://github.com/agbleze/objectscope/commit/a4b78e3decf3e291f27f8d2532ad012ac0c79cf8))

* build: update ci-cd ([`fc31cfd`](https://github.com/agbleze/objectscope/commit/fc31cfda273b82f8bcf792b0521501141ed52eb7))

* build: update poetry.lock ([`26538f3`](https://github.com/agbleze/objectscope/commit/26538f37ec07810176351ccefd923f5afa408770))

* build: remove detectron2 from dependencies ([`686f5fa`](https://github.com/agbleze/objectscope/commit/686f5fa200e219cf7ba17908a0b9e9fdf26b6f72))

* build: add git install ([`dfb543c`](https://github.com/agbleze/objectscope/commit/dfb543c45297d259092b3a724a02b54858d833f1))

* build: update packages ([`50a641d`](https://github.com/agbleze/objectscope/commit/50a641da30ba9cf653504b45fba614b8846a856d))

* build: update packages ([`e9f935a`](https://github.com/agbleze/objectscope/commit/e9f935a6e017be7092a6fa545762adc7843be173))

* build: update packages ([`f5335eb`](https://github.com/agbleze/objectscope/commit/f5335eb168028f99f7e91e63c0bc452a30d40af7))

* build: add packages ([`be6dc3b`](https://github.com/agbleze/objectscope/commit/be6dc3bab0e78405b1ed69d94179047de6141831))

* build: update workflow ([`78835c9`](https://github.com/agbleze/objectscope/commit/78835c99f35afd06aac0ae54d2d3f9050ba351b9))

### Chore

* chore: add module imports ([`8ad3940`](https://github.com/agbleze/objectscope/commit/8ad3940047f87f629814cdb5e45731321fb696d5))

### Documentation

* docs: update README ([`43a2064`](https://github.com/agbleze/objectscope/commit/43a2064cd85d6d4e58438fdcd581a94c2722e4d9))

### Feature

* feat: add fileds ([`bf0ef15`](https://github.com/agbleze/objectscope/commit/bf0ef155c6c7534ac3c0e5927a2a3609fecf66c2))

* feat: add unit tests ([`1e0a005`](https://github.com/agbleze/objectscope/commit/1e0a005feaf67ca206c5a866e7ec16c4bed8d5f7))

### Fix

* fix: remove config reading at initialization ([`d8951af`](https://github.com/agbleze/objectscope/commit/d8951afbd6fe46c65b6a03641c373d6b03adff8e))

* fix: update model_path to best_model ([`fa3a630`](https://github.com/agbleze/objectscope/commit/fa3a630947a3b3b62b35f5e3d3754180b6be65d2))

### Style

* style: remove unused imports ([`455d39a`](https://github.com/agbleze/objectscope/commit/455d39a1e1b70196bc8fb3a68f0d6e5115d8f8b3))

* style: remove unused imports ([`95ce6df`](https://github.com/agbleze/objectscope/commit/95ce6df4497fd6f53cc70b866550a42aa9eef646))

* style: remove unused imports ([`a1b3755`](https://github.com/agbleze/objectscope/commit/a1b37558f3705b1a15a4940d167ed70d5df1ff9c))

* style: remove commented code ([`014f65d`](https://github.com/agbleze/objectscope/commit/014f65d4d0e9de01fbf12528413dbe69cfa96780))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
style: update woth remote ([`a2dd59a`](https://github.com/agbleze/objectscope/commit/a2dd59a0a40b2182207cef7c42ee736951734c47))

## v1.6.1 (2025-08-12)

### Build

* build: update toml ([`f3189dd`](https://github.com/agbleze/objectscope/commit/f3189dd6cec1eda5e799f88ac11a2326733e157f))

### Fix

* fix: add creating eval_df within plot_evaluation_results ([`38f01b4`](https://github.com/agbleze/objectscope/commit/38f01b445646338b77b0ba5325464c5445429ec1))

* fix: update model evaluation ([`a017d1b`](https://github.com/agbleze/objectscope/commit/a017d1b288ffc1b4f9389c935a1bbccbbe25b6d7))

### Style

* style: remove comment ([`464b7b3`](https://github.com/agbleze/objectscope/commit/464b7b355026eedc7777e93375837d4c58c5b686))

## v1.6.0 (2025-08-11)

### Feature

* feat: add optimize ([`390aef8`](https://github.com/agbleze/objectscope/commit/390aef8de3575c67ecb437a7f35444e583f38475))

## v1.5.0 (2025-08-11)

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
remote pull to reconcil ([`2d1f90b`](https://github.com/agbleze/objectscope/commit/2d1f90be398fd20c2d075c83c43c4a1e7eced542))

## v1.4.0 (2025-08-10)

### Feature

* feat: add OnnxModelExporter ([`8db64b3`](https://github.com/agbleze/objectscope/commit/8db64b3a7232b2880e8d7de5029a9b5fb5b81e54))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
remote pull ([`241e97f`](https://github.com/agbleze/objectscope/commit/241e97f93f8905899d7dcde59c5cc43aacaaff41))

## v1.3.0 (2025-08-09)

### Feature

* feat: add run_optimize_model ([`63306b8`](https://github.com/agbleze/objectscope/commit/63306b80bb761492d10b717aa4cb7e7bb3ccc670))

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
pull remote ([`5437b1a`](https://github.com/agbleze/objectscope/commit/5437b1a41216754dafada288c0c739077560c331))

## v1.2.0 (2025-08-08)

### Build

* build: comment-out publish to pypi ([`8d012b7`](https://github.com/agbleze/objectscope/commit/8d012b79f779bdcf2f17cc88f21ed8db099643b9))

### Feature

* feat: add reading from envvar ([`b351d26`](https://github.com/agbleze/objectscope/commit/b351d2650eccd7fbcd79af12ce2d425138faf404))

* feat: add auto-opt ([`5eaec12`](https://github.com/agbleze/objectscope/commit/5eaec124f18b59841b23b42d018dcf60de26e28e))

### Fix

* fix: change cmd for optimize_model ([`c2d7c15`](https://github.com/agbleze/objectscope/commit/c2d7c15ad55d956b6b4379faeb3a87c2f1062645))

* fix: fix command auto-opt ([`aa6f79d`](https://github.com/agbleze/objectscope/commit/aa6f79d1c64815ed64a778021b698298355c56fa))

### Style

* style: add missing imports ([`9b7e4d1`](https://github.com/agbleze/objectscope/commit/9b7e4d15bbc1f3d848da95b3623e7b8cd78ea759))

## v1.1.0 (2025-08-04)

### Unknown

* Merge branch &#39;main&#39; of https://github.com/agbleze/objectscope into main
pull remote ([`361aa6d`](https://github.com/agbleze/objectscope/commit/361aa6d7de6dbbeaf2e046803f8333f9ec4511df))

## v1.0.0 (2025-08-03)

### Build

* build: comment-out tests ([`d409a21`](https://github.com/agbleze/objectscope/commit/d409a21dcdf6f5ae99f4eff30d7dda510cf135fc))

* build: update lock ([`b3b88cc`](https://github.com/agbleze/objectscope/commit/b3b88cc677be92239e6c5c72163c5d63874c16d7))

* build: update locks ([`2877060`](https://github.com/agbleze/objectscope/commit/28770600ebf6bddca397e08e7b9347359f009639))

* build: update pyproject.toml ([`2e20a31`](https://github.com/agbleze/objectscope/commit/2e20a31a27db40f9519f76377cbc23ee81f74710))

* build: add release ([`244fe16`](https://github.com/agbleze/objectscope/commit/244fe162909bb1387de4b45c0661d493ae6a1410))

* build: update python version ([`f56f5b9`](https://github.com/agbleze/objectscope/commit/f56f5b9fc14a25bf3b30d59ca1ca8f7ec2f4921a))

### Documentation

* docs: update README.md ([`d3c04e7`](https://github.com/agbleze/objectscope/commit/d3c04e785784ba0c0d24b463255ccd58d7b13dc7))

* docs: update README.md ([`823ce30`](https://github.com/agbleze/objectscope/commit/823ce306874402c2cbe8bfbb84540f0a86974c6f))

* docs: add README ([`bbb4c56`](https://github.com/agbleze/objectscope/commit/bbb4c566f61fcbe8c035900141e5cec5a019609f))

### Feature

* feat: add tensorboard ([`03e2ac4`](https://github.com/agbleze/objectscope/commit/03e2ac4b9d42ad6795361f988948c82660eb0cb7))

* feat: add launch_tensorboard ([`d2d9290`](https://github.com/agbleze/objectscope/commit/d2d9290cd3cd386f60283eab3c5dd632b279b022))

* feat: add config read ([`7f2a401`](https://github.com/agbleze/objectscope/commit/7f2a4011a075fc22708446cd16d6f2abd3990330))

* feat: add config ([`e56f9e6`](https://github.com/agbleze/objectscope/commit/e56f9e604c7788ca1640ddd86d063766a1e8b533))

* feat: add train cli ([`acf4e25`](https://github.com/agbleze/objectscope/commit/acf4e25eab7fbee7a2e8e1420caa636aea654a88))

### Fix

* fix: change name of worflow file ([`5e4e472`](https://github.com/agbleze/objectscope/commit/5e4e472e71aaa4da2885a8a4ed716c002afe05ac))

* fix: change envar read ([`acef93c`](https://github.com/agbleze/objectscope/commit/acef93c5fbbba2addbea882a759d4216a77899ac))

* fix: change __call__ to run for readability ([`42b8875`](https://github.com/agbleze/objectscope/commit/42b887548c9f3f7883913a0295b5ad60d2c73085))

* fix: resolve import from trainer ([`1e81f26`](https://github.com/agbleze/objectscope/commit/1e81f26c8e78fff4a9b5546278424fccdf7d04c5))

* fix: merge master to main ([`80caf66`](https://github.com/agbleze/objectscope/commit/80caf669efdf4cff2739b185bcbca66511b3360b))

### Unknown

* initialize ([`e66f906`](https://github.com/agbleze/objectscope/commit/e66f9064c1f9d832c82d29b2a23fc34af4d939ae))

* Initial commit ([`c4e4b36`](https://github.com/agbleze/objectscope/commit/c4e4b36ca39149d34b9b0cc9d25da70e211fff25))
