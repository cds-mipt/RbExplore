# RbExplore
## 1 Build image
```sh
cd $HOME/experiments
git clone https://github.com/ugadiarov-la-phystech-edu/rbexplore.git
cd rbexplore/docker
docker build --build-arg UNAME=$(id -un) --build-arg UID=$(id -u) --build-arg GID=$(id -g) --tag $(id -un)/rbexplore .
```
## 2 Run experiment
```sh
cd $HOME/experiments/rbexplore
# Need to copy rom.nes into custom_integrations/POP/
docker run --ipc=host --gpus "device=2" --rm --name pop_level-01_sword-start_run-0 -w /tmp/shared/ -u $(id -u):$(id -g) -v $HOME/experiments/pop/rbexplore:/tmp/shared -d $(id -un)/rbexplore:latest ./start.sh
```
## 3 Draw schema
```sh
cd $HOME/expreriments/rbexplore
python script/draw_schema.py --level_image_path script/image/level_01.png --level_description_path script/description/level_01.json --data_path script/example/graph_level_01.json --data_type graph_data --result_path clusters_level_01.png --radius 7 --color 255 140 0
```
