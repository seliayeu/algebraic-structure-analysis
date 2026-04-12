### Bandwidth Propagation Analysis


### How to run it?
Build the Docker image:

```bash
docker build -f Dockerfile.artifact -t bpa-artifact .
```

For ARM based architectures:

```bash
docker build --platform linux/amd64 -f Dockerfile.artifact -t bpa-artifact .
```

Run the container in detached mode:
```bash
docker run -d -v $(pwd)/results:/app/bpa/results --name bpa-experiments bpa-artifact:latest
```


###### Currently supported operators:

- `linalg.matmul`
- `linalg.batch_matmul`
- `linalg.elementwise`
- `linalg.transposition`

- `dia.matmul`
- `dia.batch_matmul`
- `dia.transposition`
- `dia.elementwise`

