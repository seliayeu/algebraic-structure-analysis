### Bandwidth Propagation Analysis


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
|`diagonal` | `elementwise` | `dia -> dia`| ✅ |
|`diagonal` | `elementwise` | `dense -> dense`| - |
|`diagonal` | `elementwise` | `dense,dense -> dense`| ✅ |
|`diagonal` | `elementwise` | `dense,dense -> dia`| - |
|`diagonal` | `elementwise` | `dia,dia -> dia`| ✅ |
|`diagonal` | `elementwise` | `dense,dia -> dia`| - |
|`diagonal` | `elementwise` | `dense,dia -> dense`| - |
|`diagonal` | `elementwise` | `dia,dense -> dia`| - |
|`diagonal` | `elementwise` | `dia,dense -> dense`| - |
|`diagonal` | `elementwise` | `dia,dia -> dense`| - |
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
|`banded` |` matmul` | `dia,dense -> dense`| - |
|`banded` |` matmul` | `dia,dense -> dia`| - |
|`banded` |` matmul` | `dense,dia -> dia`| ✅ |
|`banded` |` matmul` | `dense,dia -> dense`| - |
|`banded` | `transposition` | `dense`| ✅ |
|`banded` | `transposition` | `dia`| ✅ |
|`banded` | `elementwise` | `dense,dense -> dense`| ✅ |
|`banded` | `elementwise` | `dense,dense -> dia`| - |
|`banded` | `elementwise` | `dia,dia -> dia`| - |
|`banded` | `elementwise` | `dense,dia -> dia`| - |
|`banded` | `elementwise` | `dense,dia -> dense`| - |
|`banded` | `elementwise` | `dia,dense -> dia`| - |
|`banded` | `elementwise` | `dia,dense -> dense`| - |
|`banded` | `elementwise` | `dia,dia -> dense`| - |
|`banded` | `batch_matmul` | `dense,dense -> dense`| ✅ |
|`banded` | `batch_matmul` | `dense,dense -> dia`| - |
|`banded` | `batch_matmul` | `dia,dia -> dia`| - |
|`banded` | `batch_matmul` | `dense,dia -> dia`| - |
|`banded` | `batch_matmul` | `dense,dia -> dense`| - |
|`banded` | `batch_matmul` | `dia,dense -> dia`| - |
|`banded` | `batch_matmul` | `dia,dense -> dense`| - |
|`banded` | `batch_matmul` | `dia,dia -> dense`| - |

