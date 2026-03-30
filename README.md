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


###### Supported Kernels

|Type|  Operator | Formats | Status |
| ------- | ------- | -------- | -------- |
|`diagonal`| `matmul` | `dense,dense -> dense`| ✅ |
|`diagonal` | `matmul` | `dense,dense -> dia`| ✅ |
|`diagonal` | `matmul` | `dia,dia -> dia`| ✅ |
|`diagonal` | `matmul` | `dia,dense -> dia`| ✅ |
|`diagonal` | `matmul` | `dia,dense -> dense`| - |
|`diagonal` | `matmul` | `dia,dia -> dense`| N/A |
|`diagonal` | `matmul` | `dense,dia -> dia`| - |
|`diagonal` | `matmul` | `dense,dia -> dense`| - |
|`diagonal` | `transposition` | `dense`| ✅ |
|`diagonal` | `transposition` | `dia`| ✅ |
|`diagonal` | `elementwise` | `dense -> dense`| N/A |
|`diagonal` | `elementwise` | `dense,dense -> dense`| N/A |
|`diagonal` | `elementwise` | `dense,dense -> dia`| N/A |
|`diagonal` | `elementwise` | `dia,dia -> dia`| N/A |
|`diagonal` | `elementwise` | `dense,dia -> dia`| N/A |
|`diagonal` | `elementwise` | `dense,dia -> dense`| N/A |
|`diagonal` | `elementwise` | `dia,dense -> dia`| N/A |
|`diagonal` | `elementwise` | `dia,dense -> dense`| N/A |
|`diagonal` | `elementwise` | `dia,dia -> dense`| N/A |
|`diagonal` |`batch_matmul` | `dense,dense -> dense`| ✅ |
|`diagonal` | `batch_matmul` | `dense,dense -> dia`| - |
|`diagonal` | `batch_matmul` | `dia,dia -> dia`| - |
|`diagonal` | `batch_matmul` | `dense,dia -> dia`| - |
|`diagonal` | `batch_matmul` | `dense,dia -> dense`| - |
|`diagonal` | `batch_matmul` | `dia,dense -> dia`| - |
|`diagonal` | `batch_matmul` | `dia,dense -> dense`| - |
|`diagonal` | `batch_matmul` | `dia,dia -> dense`| - |
|`banded`| `matmul` | `dense,dense -> dense`| ✅ |
|`banded` | `matmul` | `dense,dense -> dia`| ✅ |
|`banded` | `matmul` | `dia,dia -> dia`| ✅ |
|`banded` |` matmul` | `dia,dia -> dense`| ✅ |
|`banded` |` matmul` | `dia,dense -> dense`| ✅ |
|`banded` |` matmul` | `dia,dense -> dia`| - |
|`banded` |` matmul` | `dense,dia -> dia`| ✅ |
|`banded` |` matmul` | `dense,dia -> dense`| - |
|`banded` | `transposition` | `dense`| ✅ |
|`banded` | `transposition` | `dia`| ✅ |
|`banded` | `elementwise` | `dia -> dia`| ✅ |
|`banded` | `elementwise` | `dense,dense -> dense`| ✅ |
|`banded` | `elementwise` | `dense,dense -> dia`| ✅ |
|`banded` | `elementwise` | `dia,dia -> dia`| ✅ |
|`banded` | `elementwise` | `dense,dia -> dia`| ✅ |
|`banded` | `elementwise` | `dense,dia -> dense`| ✅ |
|`banded` | `elementwise` | `dia,dense -> dia`| ✅ |
|`banded` | `elementwise` | `dia,dense -> dense`| ✅ |
|`banded` | `elementwise` | `dia,dia -> dense`| ✅ |
|`banded` | `batch_matmul` | `dense,dense -> dense`| ✅ |
|`banded` | `batch_matmul` | `dense,dense -> dia`| - |
|`banded` | `batch_matmul` | `dia,dia -> dia`| - |
|`banded` | `batch_matmul` | `dense,dia -> dia`| - |
|`banded` | `batch_matmul` | `dense,dia -> dense`| - |
|`banded` | `batch_matmul` | `dia,dense -> dia`| - |
|`banded` | `batch_matmul` | `dia,dense -> dense`| - |
|`banded` | `batch_matmul` | `dia,dia -> dense`| - |

