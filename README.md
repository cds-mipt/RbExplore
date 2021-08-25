# RbExplore

# Experiments
https://wandb.ai/ulaelfray/pop_rbexplore_level-1/reports/-Prince-of-Persia--Vmlldzo4NDYyOTE

# Examples
It is assumed in examples that the project's directory is `$HOME/experiments/rbexplore`.
```sh
cd $HOME/experiments
git clone https://github.com/ugadiarov-la-phystech-edu/rbexplore.git
```
## 1 Build image
```sh
cd $HOME/experiments/rbexplore/docker
docker build --build-arg UNAME=$(id -un) --build-arg UID=$(id -u) --build-arg GID=$(id -g) --tag $(id -un)/rbexplore .
```
## 2 Run experiment
```sh
cd $HOME/experiments/rbexplore
# Need to copy rom.nes into custom_integrations/POP/
docker run --ipc=host --gpus "device=2" --rm --name pop_level-01_sword-start_run-0 -w /tmp/shared/ -u $(id -u):$(id -g) -v $HOME/experiments/pop/rbexplore:/tmp/shared -d $(id -un)/rbexplore:latest ./start.sh
```
## 3 Draw schema
### 3.1 Draw clusters from graph
```sh
cd $HOME/expreriments/rbexplore
python script/draw_schema.py --level_image_path script/image/level_01.png --level_description_path script/description/level_01.json --data_path script/example/graph_level_01.json --data_type graph_data --result_path clusters_level_01.png --radius 7 --color 255 140 0
```
### 3.2 Draw visited coordinates
```sh
cd $HOME/expreriments/rbexplore
python script/draw_schema.py --level_image_path script/image/level_01.png --level_description_path script/description/level_01.json --data_path script/example/rooms_data_level_01.json --data_type rooms_data --result_path visited_coordinates_level_01.png --radius 2 --color 255 255 255
```
## 4 Compute %Cov and upload to wandb
Reference coverage data is needed in order to compute *%Cov*.
```sh
cd $HOME/expreriments/rbexplore
python script/upload_wandb.py --level_description_path script/description/level_01.json --level_ref_coverage_path script/ref_coverage/rooms_data_01.json --data_glob_path 'script/example/rooms_data*json' --project_name test-project --run_name run-0 --discretization_step 32
```
## 5 Generate reference coverage data
### 5.1 Generate reference coverage data for level 6. For every room data is stored into separate file.
```sh
cd $HOME/expreriments/rbexplore
# Need to copy rom.nes into custom_integrations/POP/
docker run --rm --name pop_level-06_generate-coverage -w /tmp/shared/ -u $(id -u):$(id -g) -v $HOME/experiments/pop/rbexplore:/tmp/shared -d $(id -un)/rbexplore:latest ./generate_coverage.sh
```
### 5.2 Combine coverage data into one file.
```sh
cd $HOME/expreriments/rbexplore
python script/combine_rooms_data.py --data_glob_path 'coverage_level_06/rooms_data*json' --result_path rooms_data_06.json
```