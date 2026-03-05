import random

class MLIRGenerator:
    def __init__(self):
        self.ssa_counter = 0
        self.lines = []
        
        self.unconsumed = []
        self.available = []
        
        self.rank = random.choice([2, 3])
        if self.rank == 2:
            self.shape = (1024, 1024)
        else:
            self.shape = (2, 1024, 1024)

        self.type_str = f"tensor<{'x'.join(map(str, self.shape))}xf32>"

    def next_ssa(self):
        name = f"%val{self.ssa_counter}"
        self.ssa_counter += 1
        return name

    def emit(self, line):
        self.lines.append(line)

    def generate_matrix_data(self, rows, cols, is_banded, upper, lower):
        mat = [[0.0 for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if is_banded and (c - r > upper or r - c > lower):
                    mat[r][c] = 0.0
                else:
                    mat[r][c] = round(random.uniform(1.0, 10.0), 2)
        return mat

    def generate_tensor_data(self, shape, is_banded, upper, lower):
        if len(shape) == 2:
            return self.generate_matrix_data(shape[0], shape[1], is_banded, upper, lower)
        elif len(shape) == 3:
            return [self.generate_matrix_data(shape[1], shape[2], is_banded, upper, lower) for _ in range(shape[0])]

    def get_new_constant(self, is_init=False):
        name = self.next_ssa()
        
        if is_init:
            self.emit(f"    {name} = arith.constant dense<0.0> : {self.type_str}")
            return name
            
        is_banded = random.random() >= 0.5
        dim0 = len(self.shape) - 2
        dim1 = len(self.shape) - 1
        dim_row = self.shape[dim0]
        dim_col = self.shape[dim1]
        
        upper = random.randint(0, dim_col - 1)
        lower = random.randint(0, dim_row - 1)
        
        attr_str = ""
        if is_banded:
            attr_str = f" {{metadata = {{lowerBw = {lower} : i64, propertyDims = [{dim0}, {dim1}], upperBw = {upper} : i64}}}}"
            
        data = self.generate_tensor_data(self.shape, is_banded, upper, lower)
        self.emit(f"    {name} = arith.constant{attr_str} dense<{str(data)}> : {self.type_str}")
        
        self.available.append(name)
        return name

    def get_inputs(self, count):
        inputs = []
        for _ in range(count):
            if self.unconsumed and random.random() < 0.8:
                idx = random.randrange(len(self.unconsumed))
                val = self.unconsumed.pop(idx)
                inputs.append(val)
            elif self.available and random.random() < 0.5:
                inputs.append(random.choice(self.available))
            else:
                inputs.append(self.get_new_constant())
        return inputs

    def gen_op(self, op_type):
        lhs, rhs = self.get_inputs(2)
        init = self.get_new_constant(is_init=True)
        res = self.next_ssa()
        
        if op_type == "matmul":
            op_name = "linalg.matmul" if self.rank == 2 else "linalg.batch_matmul"
        else:
            op_name = f"linalg.{op_type}"
            
        self.emit(f"    {res} = {op_name} ins({lhs}, {rhs} : {self.type_str}, {self.type_str}) outs({init} : {self.type_str}) -> {self.type_str}")
        
        self.unconsumed.append(res)
        self.available.append(res)

    def generate(self, num_ops=15):
        for _ in range(num_ops):
            op_type = random.choices(["matmul", "add", "mul"], weights=[0.4, 0.3, 0.3])[0]
            self.gen_op(op_type)

        while len(self.unconsumed) > 1:
            lhs = self.unconsumed.pop(0)
            rhs = self.unconsumed.pop(0)
            init = self.get_new_constant(is_init=True)
            res = self.next_ssa()
            
            op_choice = random.choice(["matmul", "add", "mul"])
            if op_choice == "matmul":
                op_name = "linalg.matmul" if self.rank == 2 else "linalg.batch_matmul"
            else:
                op_name = f"linalg.{op_choice}"
            
            self.emit(f"    {res} = {op_name} ins({lhs}, {rhs} : {self.type_str}, {self.type_str}) outs({init} : {self.type_str}) -> {self.type_str}")
            
            self.unconsumed.append(res)
            self.available.append(res)

        final_val = self.unconsumed[0] if self.unconsumed else self.get_new_constant()

        output = [
            f"func.func @random_linalg_graph() -> {self.type_str} {{",
        ]
        output.extend(self.lines)
        output.append(f"    return {final_val} : {self.type_str}")
        output.append("}")
        return "\n".join(output)

if __name__ == "__main__":
    gen = MLIRGenerator()
    mlir_code = gen.generate(10)
    with open("out.mlir", "w") as file:
        file.write(mlir_code)
        
