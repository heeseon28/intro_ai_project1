# Windows의 VMMRdb를 GPU 서버로 옮기기

코드는 Git으로 공유하고, 원본 이미지는 별도 파일로 전송합니다. 이 도구는 기존 `splits/train.csv`와 `splits/val.csv`에 있는 **100,700장**만 하나의 ZIP으로 묶습니다. CompCar 및 테스트 이미지는 포함하지 않으며 원본 이미지와 분할 CSV는 변경하지 않습니다.

Python 표준 라이브러리만 사용합니다. JPEG를 다시 압축하지 않고 묶으므로 Windows에는 선택 이미지 용량만큼 추가 공간이 필요합니다. 시작할 때 필요한 용량을 출력하고, 공간 부족 시 중단합니다. 모든 이미지의 MD5를 기존 CSV와 대조하며 2,000장마다 진행률을 표시합니다. 실패한 작업은 완성된 ZIP이나 데이터 폴더로 표시하지 않습니다.

## 1. Windows PC: ZIP 만들기

Windows VS Code에서 프로젝트 루트의 PowerShell 터미널을 사용합니다. 먼저 새 코드를 가져옵니다. 이전 `preflight` 실행으로 변경된 결과 파일과 충돌한다면 그 변경을 보관한 뒤 pull해야 합니다.

```powershell
git pull --ff-only origin main
python task2_preliminary_seojin/transfer_data.py pack
```

기본 데이터 경로는 `dataset_paths.json`에 저장된 Windows 경로입니다. 실제 위치가 달라졌다면 `--data-root "D:\datasets\VMMRdb"`를 붙입니다.

`PACK_COMPLETE`가 나오면 프로젝트 안에 아래 파일이 생성됩니다.

```text
dataset/transfer/vmmrdb_task2_train_val.zip
```

ZIP 생성 중 원본 이미지를 이동하거나 수정하지 마세요. 같은 이름의 완성된 ZIP이 있으면 덮어쓰지 않습니다. 파일이 이미 정상 생성되었다면 그 파일을 전송하면 됩니다.

## 2. 서버로 ZIP 전송

서버의 수신 위치:

```text
/home/work/workspace/intro_ai_project1/dataset/transfer/vmmrdb_task2_train_val.zip
```

**실제 접속 방식부터 확인해야 합니다.** 아래 SSH 예시는 `school-server`라는 SSH 별칭을 사용하고, 그 SSH 세션에서 위 서버 경로에 접근할 수 있을 때만 그대로 사용할 수 있습니다. 컨테이너 밖의 호스트로 접속하는 경우 경로가 다를 수 있습니다.

```powershell
# Windows PowerShell: school-server를 실제 SSH 별칭으로 바꾼 뒤 사용
ssh school-server "mkdir -p /home/work/workspace/intro_ai_project1/dataset/transfer"
scp .\dataset\transfer\vmmrdb_task2_train_val.zip school-server:/home/work/workspace/intro_ai_project1/dataset/transfer/
```

별칭 대신 `계정@호스트`를 사용할 수 있습니다. 별도 포트가 필요하면 SSH는 `-p 포트`, SCP는 `-P 포트`를 지정합니다. 웹 접속만 지원하는 서버는 해당 서비스의 파일 업로드 기능이나 관리자가 제공한 전송 경로를 사용해야 합니다.

전송 완료 후 ZIP 파일의 정확한 서버 위치를 확인합니다. `dataset/`은 이미 Git에서 제외되어 있으며 이미지를 Git에 올릴 필요가 없습니다.

## 3. GPU 서버: 풀기와 검사

다음 명령은 서버의 프로젝트 루트에서 실행합니다. ZIP을 다른 위치에 올렸다면 `unpack --archive /실제/업로드/파일.zip`으로 지정합니다.

```bash
python3 task2_preliminary_seojin/transfer_data.py unpack
python3 task2_preliminary_seojin/preflight.py --data-root dataset/VMMRdb
python3 task2_preliminary_seojin/train_baseline.py --data-root dataset/VMMRdb --device cuda --smoke --run-name baseline_seed0_seojin_server
```

`unpack`은 서버와 Windows의 분할 CSV SHA256이 같은지, 이미지가 모두 있는지, 각 이미지의 MD5가 맞는지 확인합니다. 테스트 이미지는 요구하지 않습니다. 기존 `dataset/VMMRdb` 폴더가 있으면 보존하고 중단합니다.

## 4. GPU 서버: 실제 학습

smoke가 완료되면 다음 명령으로 12 epoch 실험을 실행합니다. 서버 기본 batch 64로 먼저 확인하며, 메모리 부족이 나면 원인을 확인한 뒤 설정 변경 여부를 결정합니다.

```bash
python3 task2_preliminary_seojin/train_baseline.py --data-root dataset/VMMRdb --device cuda --run-name baseline_seed0_seojin_server
```

접속 종료에도 학습을 유지하려면 서버 터미널에서 다음과 같이 백그라운드로 실행할 수 있습니다. 위 일반 실행과 동시에 중복 실행하지 않습니다.

```bash
mkdir -p task2_preliminary_seojin/results
nohup python3 -u task2_preliminary_seojin/train_baseline.py --data-root dataset/VMMRdb --device cuda --run-name baseline_seed0_seojin_server > task2_preliminary_seojin/results/server_console.log 2>&1 < /dev/null &
```

결과는 `task2_preliminary_seojin/results/baseline_seed0_seojin_server/`에 저장됩니다. 이 폴더의 `run.log`와 `status.json`으로 진행 상태를 확인합니다. 학습을 중단한 뒤 재개하려면 같은 명령에 `--resume`을 붙입니다. 아직 학습 중인 실행에 `--resume`을 중복으로 실행하지 마세요.

```bash
tail -n 20 task2_preliminary_seojin/results/baseline_seed0_seojin_server/run.log
```
