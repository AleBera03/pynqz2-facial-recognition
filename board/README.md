# Board

## Dexplorer
**DExplorer** is a tool on board which provides DPU running mode configurations, DNNDK version checking, DPU status checking, etc.
- `dexplorer -s`: display status info of DPU core, such as running mode (normal or debug)
- `dexplorer -w`: display info about DPU core, such as supported layers

## opencv-info
Inside the `opencv-info` folder you can find a script which displays several pieces of information about the OpenCV build on the current PetaLinux image. Just run:
```
./cvinfo
```

## `zynq7020_dnndk_v3.1`
If some tool seems not existent, try to reinstall packages and dnndk runtime
```
cd zynq7020_dnndk_v3.1
./install.sh
```

## Project structure example
```
facial_recognition
│   .gitignore
│   face_database.csv
│   main
│   Makefile
│   README.md
│
├───images
│   ├───0
│   │       0.jpg
│   │       1.jpg
│   │       10.jpg
│   │       11.jpg
│   │       2.jpg
│   │       3.jpg
│   │       4.jpg
│   │       5.jpg
│   │       6.jpg
│   │       7.jpg
│   │       8.jpg
│   │       9.jpg
│   │
│   └───1
│           0.jpg
│
├───models
│   ├───detector
│   │       haarcascade_frontalface_alt2.xml
│   │       lbpcascade_frontalface.xml
│   │
│   └───recognizer
│           dpu_face_resnet50.elf
│
├───objects
│       db.o
│       det.o
│       main.o
│       rec.o
│       server.o
│
├───people
│   └───me
│           fototessera.jpg
│
└───src
        db.cpp
        db.h
        det.cpp
        det.h
        main.cpp
        rec.cpp
        rec.h
        httlib.h
```

### Database
The database is a `.csv` file with the following format:
```
<id>, <name>, <emb_1>, <emb_2>, ... ,<emb_2048>
```
The 2048-D embedding vector contains 2048 floats computed by the compiled ResNet50 model (pre-trained on VGGFace2) on a cropped photo where a face has been detected.

For each person/identity there could be more than one embedding vector. By default, if a person is enrolled via live camera, there are 12 embedding vectors.

### Makefile
To recompile the entire program, run:
```
cd path/to/facial_recognition
make clean  # remove all obj files and executables
make        # compile all with -O2 flag
```
If you want debug mode, run:
```
cd path/to/facial_recognition
make clean
make debug
```
Both produce an executable with the same name (`main`).


### Images
Folder which contains every cropped photo per person. Each subfolder uses an id as its name, related to the person id. It is useful during debug for understanding whether saved photos have enough quality.

### Models
There are 2 detectors:
- **LBP**: used for faster streaming — more fps, but it can produce more false positives and is sensitive to illumination
- **HAAR**: more accurate and resilient under hard illumination, but slower (2–3 fps less)

Inside the `recognizer` folder there is the compiled ResNet50 model.

### People
It is possible to save one or more photos per person. For enrollment, use `enroll <id> <name> -i|--input <path>`.


### Source

- `main.cpp`: contains the main loop, thread invocation and CLI handling
- `det`: detector implementation. To change mode (optimized or not), modify the `OPTIMIZED` variable in `det.h` and recompile. `OPTIMIZED` mode:
  - use LBP classifier, faster but less accurate
  - does not execute [Histogram Equalization](https://en.wikipedia.org/wiki/Histogram_equalization) (due to poor compatibility of LBP to `equalizeHist` opencv function)
- `rec`: recognition implementation
- [`httlib`](https://github.com/yhirose/cpp-httplib): only-header library which aims to expose a small and reliable streaming server
- `db`: database interface which can add or remove a row given an id


## Usage

Go inside the `facial_recognition` folder and run:
```
cd path/to/facial_recognition
./main
```

It will open the following prompt:
```
root@pynqz2_dpu:~/facial_recognition# ./main
[OK] Classifier Loaded

(main:17839): GStreamer-CRITICAL **: 10:00:22.967: gst_element_get_state: assertion 'GST_IS_ELEMENT (element)' failed
[OK] created socket
[OK] port binding 8080
[REC] DPU task ready — kernel: face_resnet50 | embedding: 2048-D
[DB] Loaded 2 rows from face_database.csv

[CMD] Commands:
  enroll <id> <name> [OPTIONS]     — enroll new person/identity
  OPTIONS
         -i|--input <folder_path>
  rec                              — toggle recognition
  show                             — toggle streaming HTTP (client connection)
  list                             — show list of identities within DB
  remove <id>                      — remove person/identity from DB
  quit                             — close program
  help                             — show this message

>

```

### `enroll`
This command registers a new person in the database. If the identity already exists, the program will ask whether you want to overwrite the entire class or not.
A new subfolder inside `images` will be created with all cropped photos.
Note that recognition is based on database embeddings, not on the raw photos.

> **WARNING**
> Program has a feature that let user choose even a single photo (one-shot) to enroll a new identity. This use case doesn't match project architecture. Due to statistics classifier such as SVM or NCC, it is really hard get satisfying results. Future work could use a [siemese network](https://en.wikipedia.org/wiki/Siamese_neural_network)


### `rec`
Toggle recognition. A green _REC:ON_ label on the top-right will be printed if active.

Every 5 frames a new recognition attempt is made. If the SVM does not classify any face, `unknown` is printed.

### `show`
Toggle streaming availability.
Working mechanism follows concurrency pattern
#### From off to on
- when streaming is off, `set_chunked_content_provider` performs a [condition variable](https://en.cppreference.com/cpp/thread/condition_variable) waiting unless someone (re)starts streaming
- if user press `show`, a boolean variable will be `true` unlocking waiting and content provider routine will continue
#### From on to off
- when streaming is on, `set_chunked_content_provider` normally works, but user press `show`. Therefore, condition variable notify thread so that content provider routine will continue
- By continuing, a condition will notice that streaming is now off. So, server will stop providing data and it will return to waiting that user press again `show`


