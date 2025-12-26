# Implementation Checklist

## Overview

The GPU simulation is a compute shader-driven mode designed for massive-scale real-time organism ecosystems. It leverages GPU parallel processing to handle 100K-200K cells with full fluid, chemical, and light simulation.

**Core Requirements:**

- **Cell Count:** 100K-200K cells (user configurable)
- **Performance Target:** 100K cells at >50tps = <0.2Fr (<20ms per tick)
- **Execution:** GPU compute shaders (WGSL)
- **Control:** Variable speed controls, no backwards simulation
- **Determinism:** As close as possible to CPU/Preview (acceptable variance)
- **Features:** Full fluid and gas simulation, chemicals, light, all cell types
- **Same Physics:** Results similar to CPU and Preview modes

---

## Table of Contents

1. [[#GPU Architecture]]
2. [[#Buffer Management]]
3. [[#Compute Shader Pipeline]]
4. [[#Physics Systems]]
5. [[#Fluid Simulation]]
6. [[#Environmental Systems]]
7. [[#Cell Types Implementation]]
8. [[#GPU-Driven Rendering]]
9. [[#Performance Optimization]]
10. [[#Testing & Validation]]

---

## GPU Architecture

### Compute Shader Overview

- [ ] WGSL shader language
- [ ] Bevy + WGPU integration
- [ ] Compute pipeline setup
- [ ] Binding group management
- [ ] Shader compilation and validation
- [ ] Error handling and debugging

### GPU Memory Model

- [ ] Understand GPU memory hierarchy:
    - [ ] Global memory (slowest, largest)
    - [ ] Shared memory / workgroup memory (fast, limited)
    - [ ] Registers (fastest, very limited)
- [ ] Memory coalescing for efficient access
- [ ] Bank conflicts avoidance
- [ ] Alignment requirements (16-byte boundaries)

### Workgroup Configuration

- [ ] Determine optimal workgroup size (64, 128, 256 threads)
- [ ] Balance occupancy vs resource usage
- [ ] Workgroup dimensions (1D, 2D, 3D)
- [ ] Thread indexing and dispatch calculations
- [ ] Workgroup local memory usage

### Execution Model

- [ ] Compute shader dispatch flow:
    1. [ ] CPU prepares buffers
    2. [ ] CPU dispatches compute shader
    3. [ ] GPU executes shader (parallel)
    4. [ ] CPU waits for completion (barrier)
    5. [ ] Next shader or readback results
- [ ] Pipeline barriers between shaders
- [ ] Synchronization within workgroups
- [ ] No synchronization across workgroups

### GPU Hardware Considerations

- [ ] Target minimum GPU spec (define requirements)
- [ ] Shader model version
- [ ] Maximum buffer sizes
- [ ] Maximum workgroup size
- [ ] Compute capability checks
- [ ] Fallback for unsupported hardware

---

## Buffer Management

### Cell Data Buffers (Structure of Arrays)

#### Primary Cell Buffers

- [ ] Position buffer (`array<vec3<f32>, 200000>`)
- [ ] Velocity buffer (`array<vec3<f32>, 200000>`)
- [ ] Rotation buffer (`array<vec4<f32>, 200000>` - quaternions)
- [ ] Angular velocity buffer (`array<vec3<f32>, 200000>`)
- [ ] Force accumulator buffer (`array<vec3<f32>, 200000>`)
- [ ] Torque accumulator buffer (`array<vec3<f32>, 200000>`)
- [ ] Mass buffer (`array<f32, 200000>`)
- [ ] Radius buffer (`array<f32, 200000>`)
- [ ] Age buffer (`array<f32, 200000>`)
- [ ] Split count buffer (`array<u32, 200000>`)
- [ ] Genome index buffer (`array<u32, 200000>`)
- [ ] Mode index buffer (`array<u32, 200000>`)
- [ ] Cell type buffer (`array<u32, 200000>`)
- [ ] Active flag buffer (`array<u32, 200000>` - bitpacked)
- [ ] Cell count buffer (`atomic<u32>` - single value)

#### Buffer Layout Considerations

- [ ] 16-byte alignment for vec3 (pad to vec4 or use careful packing)
- [ ] Struct padding rules in WGSL
- [ ] Memory coalescing (sequential threads access sequential memory)
- [ ] Storage buffer vs uniform buffer (storage for large data)

### Adhesion Buffers

#### Adhesion Data

- [ ] Adhesion buffer (large array, e.g., 1,000,000 capacity)
- [ ] Adhesion fields:
    - [ ] Cell A index (`u32`)
    - [ ] Cell B index (`u32`)
    - [ ] Anchor A (`vec3<f32>`)
    - [ ] Anchor B (`vec3<f32>`)
    - [ ] Rest length (`f32`)
    - [ ] Stiffness (`f32`)
    - [ ] Damping coefficient (`f32`)
    - [ ] Orientation stiffness (`f32`)
    - [ ] Orientation damping (`f32`)
    - [ ] Twist stiffness (`f32`, optional)
    - [ ] Twist damping (`f32`, optional)
    - [ ] Twist reference (`vec4<f32>`, optional quaternions)
    - [ ] Original side (`u32`)
    - [ ] Active flag (`u32`)
- [ ] Adhesion count buffer (`atomic<u32>`)

### Genome Buffers

#### Genome Library

- [ ] Genome deduplication table (complex on GPU)
- [ ] Strategy: Pre-upload unique genomes from CPU
- [ ] Genome buffer (array of all unique genomes)
- [ ] Mode data buffer (120 modes × genome count)
- [ ] Each mode contains all parameters (flat structure)
- [ ] Genome hash to index mapping (on CPU)

#### Mode Data Structure

- [ ] Pack mode data efficiently:
    - [ ] Cell type, child modes, split parameters
    - [ ] Angles (pitch/yaw stored as compact format)
    - [ ] Color (RGB packed into u32)
    - [ ] Flags (bitpacked: make adhesion, keep adhesion, prioritize, etc.)
    - [ ] Numerical parameters (mass, ratio, interval, priority, etc.)
- [ ] Total: aim for <256 bytes per mode
- [ ] 120 modes × 256 bytes = 30KB per genome

### Spatial Grid Buffers

#### Grid Data

- [ ] Grid cell buffer (200³ or adaptive size)
- [ ] Each grid cell contains:
    - [ ] Cell index list (variable length - challenging on GPU)
    - [ ] Or: Fixed-size cell index array per grid cell (wasteful but simple)
- [ ] Grid metadata buffer (dimensions, cell size, bounds)
- [ ] Grid counter buffer (cells per grid cell)

#### Grid Implementation Strategies

- [ ] **Option A: Fixed-size arrays per grid cell**
    - [ ] Each grid cell has array[10] of cell indices
    - [ ] Simple but wasteful (many empty slots)
    - [ ] 200³ × 10 × 4 bytes = 320MB
- [ ] **Option B: Compact grid with indirection**
    - [ ] Grid cell stores (start_index, count)
    - [ ] Compacted cell index list
    - [ ] Requires prefix-sum for compaction
- [ ] **Option C: Hash grid**
    - [ ] Cells hash into grid
    - [ ] Handle collisions (linear probing, chaining)
    - [ ] More compact for sparse distributions

### Fluid Grid Buffers (64³ = 262,144 cells)

#### Per-Grid-Cell Data

- [ ] Velocity buffer (`array<vec3<f32>, 262144>`)
- [ ] Velocity temporary buffer (for advection/diffusion)
- [ ] Pressure buffer (`array<f32, 262144>`)
- [ ] Divergence buffer (`array<f32, 262144>`)
- [ ] Dissolved nutrients buffer (`array<f32, 262144>`)
- [ ] Chemical concentration buffers (`array<f32, 262144>` × 4 chemicals)
- [ ] Light intensity buffer (`array<f32, 262144>`)
- [ ] Temperature buffer (`array<f32, 262144>`, optional)
- [ ] Gas concentration buffers (O2, CO2, etc.)

#### Grid Update Flags

- [ ] Active cell flags (for adaptive updates)
- [ ] Threshold values for activation
- [ ] Double buffering for read/write

### Allocation Buffers (for Cell Division)

#### Prefix-Sum for Deterministic Allocation

- [ ] Division request flags (`array<u32, 200000>`)
- [ ] Prefix-sum result (`array<u32, 200000>`)
- [ ] Total division count (`atomic<u32>`)
- [ ] Allocated indices buffer
- [ ] Scratch buffers for parallel prefix-sum

#### Compaction Buffers

- [ ] Active cell compaction (remove dead cells)
- [ ] Adhesion compaction (remove inactive adhesions)
- [ ] Stream compaction using prefix-sum

### Double/Triple Buffering

#### Why Triple Buffering

- [ ] GPU operations are asynchronous
- [ ] Avoid stalling waiting for GPU
- [ ] CPU prepares next frame while GPU works
- [ ] Three states: CPU writing, GPU reading, GPU writing

#### Buffer Sets

- [ ] Set A: Current simulation state
- [ ] Set B: Next simulation state
- [ ] Set C: Rendering snapshot
- [ ] Rotate buffers each frame
- [ ] Careful index management

### Buffer Binding Groups

- [ ] Organize buffers into binding groups
- [ ] Group 0: Cell data (positions, velocities, etc.)
- [ ] Group 1: Adhesion data
- [ ] Group 2: Grid data (spatial and fluid)
- [ ] Group 3: Genome data
- [ ] Group 4: Temporary/scratch buffers
- [ ] Minimize binding group changes for performance

---

## Compute Shader Pipeline

### Shader Organization

#### Physics Shaders

- [ ] `clear_forces.wgsl` - Reset force/torque accumulators
- [ ] `spatial_grid_rebuild.wgsl` - Rebuild spatial grid from cell positions
- [ ] `collision_detect.wgsl` - Find collision pairs using grid
- [ ] `collision_response.wgsl` - Calculate collision forces
- [ ] `boundary_forces.wgsl` - Calculate world boundary forces
- [ ] `adhesion_forces.wgsl` - Calculate adhesion forces and torques
- [ ] `integrate.wgsl` - Verlet integration step
- [ ] `cell_division_detect.wgsl` - Mark cells ready to divide
- [ ] `cell_division_allocate.wgsl` - Prefix-sum for allocation
- [ ] `cell_division_execute.wgsl` - Perform divisions
- [ ] `adhesion_inheritance.wgsl` - Create adhesions from division
- [ ] `mass_transfer.wgsl` - Transfer mass between adhered cells
- [ ] `cell_death.wgsl` - Mark and remove dead cells

#### Fluid Shaders

- [ ] `fluid_advection.wgsl` - Advect velocity field
- [ ] `fluid_divergence.wgsl` - Compute velocity divergence
- [ ] `fluid_pressure_jacobi.wgsl` - Iterative pressure solve
- [ ] `fluid_pressure_gradient.wgsl` - Apply pressure gradient
- [ ] `fluid_cell_coupling.wgsl` - Cells affect fluid
- [ ] `fluid_to_cell_forces.wgsl` - Fluid affects cells
- [ ] `chemical_diffusion.wgsl` - Diffuse chemicals in grid
- [ ] `chemical_advection.wgsl` - Advect chemicals with fluid

#### Environmental Shaders

- [ ] `light_propagation.wgsl` - Calculate light through grid
- [ ] `nutrient_diffusion.wgsl` - Diffuse dissolved nutrients
- [ ] `gas_diffusion.wgsl` - Diffuse gases

#### Cell Type Shaders

- [ ] `cell_type_update.wgsl` - Update all cell types
- [ ] Or separate shader per cell type for modularity
- [ ] Sensor calculations (Oculocyte, Senseocyte, etc.)
- [ ] Active cell actions (Flagellocyte, Myocyte, etc.)

#### Utility Shaders

- [ ] `prefix_sum.wgsl` - Parallel prefix-sum (scan) algorithm
- [ ] `compact.wgsl` - Stream compaction
- [ ] `copy_buffer.wgsl` - Buffer copies (if needed)

### Shader Execution Order (Each Tick)

#### Phase 1: Preparation

1. [ ] Clear force/torque accumulators
2. [ ] Rebuild spatial grid
3. [ ] Cell type updates (sensors read environment)

#### Phase 2: Force Calculation

4. [ ] Collision detection
5. [ ] Collision response forces
6. [ ] Boundary forces
7. [ ] Adhesion forces (main workload)
8. [ ] Fluid-to-cell forces
9. [ ] Active cell forces (Flagellocyte propulsion, etc.)

#### Phase 3: Integration

10. [ ] Verlet integration (update positions, velocities)
11. [ ] Update cell ages
12. [ ] Update cell radii from mass

#### Phase 4: Cell Division

13. [ ] Detect cells ready to divide
14. [ ] Prefix-sum for allocation
15. [ ] Execute divisions (create children)
16. [ ] Create inherited adhesions
17. [ ] Update adhesion indices

#### Phase 5: Energy and Death

18. [ ] Mass accumulation (Photocytes, Chemocytes, etc.)
19. [ ] Mass transfer between cells
20. [ ] Cell death detection
21. [ ] Compact dead cells (periodic)

#### Phase 6: Fluid Simulation

22. [ ] Advect velocity field
23. [ ] Cells affect fluid (obstacles, perturbations)
24. [ ] Compute divergence
25. [ ] Pressure solve (Jacobi iterations)
26. [ ] Apply pressure gradient (incompressibility)

#### Phase 7: Environmental Updates

27. [ ] Chemical diffusion
28. [ ] Chemical advection with fluid
29. [ ] Nutrient diffusion
30. [ ] Light propagation
31. [ ] Gas diffusion

#### Phase 8: Rendering Prep

32. [ ] Copy data to render buffers (if needed)
33. [ ] Generate instance data for rendering

### Pipeline Barriers

- [ ] Insert barriers between dependent shaders
- [ ] Ensure writes complete before reads
- [ ] Minimize barriers for performance
- [ ] Use buffer barriers, not full pipeline barriers

### Dispatch Calculations

- [ ] Calculate dispatch size from buffer size and workgroup size
- [ ] Example: `dispatch_x = ceil(cell_count / workgroup_size)`
- [ ] Handle cases where buffer not multiple of workgroup size
- [ ] Bounds checking in shaders

---

## Physics Systems

### Velocity Verlet Integration (GPU)

#### Clear Forces Shader

```wgsl
@compute @workgroup_size(256)
fn clear_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count) { return; }
    forces[idx] = vec3(0.0);
    torques[idx] = vec3(0.0);
}
```

- [ ] Implement clear forces shader
- [ ] Dispatch: `ceil(cell_count / 256)`

#### Integration Shader

```wgsl
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || !active[idx]) { return; }
    
    // Update velocity: v += (F/m) * dt
    let acceleration = forces[idx] / masses[idx];
    velocities[idx] += acceleration * dt;
    
    // Clamp velocity
    velocities[idx] = clamp_vec3(velocities[idx], max_velocity);
    
    // Update position: pos += v * dt
    positions[idx] += velocities[idx] * dt;
    
    // Angular integration
    let alpha = torques[idx] / inertia[idx];
    angular_velocities[idx] += alpha * dt;
    angular_velocities[idx] = clamp_vec3(angular_velocities[idx], max_angular_vel);
    
    // Update rotation (quaternion integration)
    rotations[idx] = integrate_quaternion(rotations[idx], angular_velocities[idx], dt);
    
    // Update age
    ages[idx] += dt;
    
    // Update radius from mass
    radii[idx] = calculate_radius(masses[idx]);
}
```

- [ ] Implement integration shader
- [ ] Quaternion integration function
- [ ] Velocity/angular velocity clamping
- [ ] Radius calculation from mass

### Spatial Grid (GPU)

#### Grid Rebuild Shader

- [ ] **Challenge:** Variable-length lists difficult on GPU
- [ ] **Approach A: Fixed-size lists**
    - [ ] Each grid cell has fixed capacity (e.g., 10 cells)
    - [ ] Use atomic counter per grid cell
    - [ ] If full, overflow cells ignored (not ideal)
- [ ] **Approach B: Two-pass compaction**
    - [ ] Pass 1: Count cells per grid cell
    - [ ] Prefix-sum to get start indices
    - [ ] Pass 2: Write cells to compacted array
    - [ ] More complex but efficient

```wgsl
@compute @workgroup_size(256)
fn grid_rebuild(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || !active[idx]) { return; }
    
    // Calculate grid cell
    let grid_pos = position_to_grid(positions[idx]);
    let grid_idx = grid_index(grid_pos);
    
    // Atomic increment grid cell counter
    let slot = atomicAdd(&grid_counts[grid_idx], 1u);
    
    // If slot < capacity, store cell index
    if (slot < GRID_CELL_CAPACITY) {
        grid_cells[grid_idx * GRID_CELL_CAPACITY + slot] = idx;
    }
}
```

- [ ] Implement grid rebuild shader
- [ ] Grid cell calculation from position
- [ ] Atomic operations for thread safety
- [ ] Handle grid cell overflow

### Collision Detection (GPU)

#### Collision Pair Generation

- [ ] Iterate over grid cells
- [ ] For each grid cell, check cells within and neighbors
- [ ] Generate collision pairs (store or process immediately)

```wgsl
@compute @workgroup_size(64)
fn collision_detect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.x;
    if (grid_idx >= grid_size) { return; }
    
    // Get cells in this grid cell
    let count = grid_counts[grid_idx];
    
    // Check all pairs in this cell
    for (var i = 0u; i < count; i++) {
        let cell_a = grid_cells[grid_idx * GRID_CELL_CAPACITY + i];
        for (var j = i + 1u; j < count; j++) {
            let cell_b = grid_cells[grid_idx * GRID_CELL_CAPACITY + j];
            check_collision(cell_a, cell_b);
        }
    }
    
    // Check neighboring grid cells
    // ... (iterate over 26 neighbors in 3D)
}

fn check_collision(cell_a: u32, cell_b: u32) {
    let pos_a = positions[cell_a];
    let pos_b = positions[cell_b];
    let distance = length(pos_b - pos_a);
    let overlap = (radii[cell_a] + radii[cell_b]) - distance;
    
    if (overlap > 0.0) {
        // Calculate collision force
        let direction = normalize(pos_b - pos_a);
        let force = stiffness * overlap * direction;
        
        // Apply damping
        let rel_vel = velocities[cell_b] - velocities[cell_a];
        let damping_force = damping * dot(rel_vel, direction) * direction;
        
        let total_force = force - damping_force;
        
        // Atomically add forces (or use force accumulators)
        atomicAdd(&forces[cell_a], -total_force);
        atomicAdd(&forces[cell_b], total_force);
    }
}
```

- [ ] Implement collision detection shader
- [ ] Neighbor grid cell iteration (3×3×3 = 27 cells)
- [ ] Collision force calculation
- [ ] Atomic force accumulation (or per-thread buffers)
- [ ] Handle large penetrations

### Adhesion Forces (GPU)

#### Adhesion Force Shader

- [ ] Iterate over all adhesions in parallel
- [ ] Calculate linear spring, damping, orientation torques
- [ ] Accumulate forces/torques to cells

```wgsl
@compute @workgroup_size(256)
fn adhesion_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= adhesion_count || !adhesion_active[idx]) { return; }
    
    let cell_a = adhesion_cell_a[idx];
    let cell_b = adhesion_cell_b[idx];
    
    // Transform anchors to world space
    let anchor_a_world = rotate_vector(rotations[cell_a], anchor_a[idx]);
    let anchor_b_world = rotate_vector(rotations[cell_b], anchor_b[idx]);
    
    // Calculate positions of anchor points
    let pos_a = positions[cell_a] + anchor_a_world * radii[cell_a];
    let pos_b = positions[cell_b] + anchor_b_world * radii[cell_b];
    
    // Linear spring force
    let direction = pos_b - pos_a;
    let distance = length(direction);
    let spring_force = stiffness[idx] * (distance - rest_length[idx]) * normalize(direction);
    
    // Linear damping
    let rel_vel = velocities[cell_b] - velocities[cell_a];
    let damping_force = damping[idx] * dot(rel_vel, normalize(direction)) * normalize(direction);
    
    let linear_force = spring_force + damping_force;
    
    // Accumulate forces
    atomicAdd(&forces[cell_a], linear_force);
    atomicAdd(&forces[cell_b], -linear_force);
    
    // Orientation torques
    // ... (similar calculations for angular corrections)
    
    // Tangential forces (from torques)
    // ... (convert torques to forces for shape maintenance)
}
```

- [ ] Implement adhesion force shader
- [ ] Linear spring and damping
- [ ] Orientation correction torques
- [ ] Twist constraints (optional)
- [ ] Tangential force calculation
- [ ] Atomic force/torque accumulation

### Cell Division (GPU)

#### Division Detection

```wgsl
@compute @workgroup_size(256)
fn division_detect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || !active[idx]) { return; }
    
    // Get mode data
    let mode = get_mode(genome_index[idx], mode_index[idx]);
    
    // Check division criteria
    let mass_ready = masses[idx] >= mode.split_mass;
    let age_ready = ages[idx] >= mode.split_interval;
    let adhesion_count = count_adhesions(idx);
    let adhesion_ready = adhesion_count >= mode.min_adhesions && 
                         adhesion_count < mode.max_adhesions;
    let split_ready = splits[idx] < mode.max_splits || mode.max_splits == -1;
    
    if (mass_ready && age_ready && adhesion_ready && split_ready) {
        division_flags[idx] = 1u;
    } else {
        division_flags[idx] = 0u;
    }
}
```

- [ ] Implement division detection shader
- [ ] Check all division criteria
- [ ] Count adhesions per cell (requires preprocessing)
- [ ] Set division flags

#### Prefix-Sum for Allocation

- [ ] Implement parallel prefix-sum (scan) algorithm
- [ ] Kogge-Stone or Blelloch scan
- [ ] Input: division_flags array
- [ ] Output: prefix-sum array (allocation indices)
- [ ] Total divisions from last element

```wgsl
// Simplified prefix-sum (actual implementation more complex)
@compute @workgroup_size(256)
fn prefix_sum_step(/* ... */) {
    // Use workgroup shared memory for local scan
    // Multiple passes for global scan
    // See GPU Gems 3 Chapter 39 for details
}
```

- [ ] Implement efficient parallel scan
- [ ] Use shared memory for workgroup-level scan
- [ ] Combine workgroup results for global scan
- [ ] Handle large arrays (>1 workgroup)

#### Division Execution

```wgsl
@compute @workgroup_size(256)
fn division_execute(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let parent_idx = global_id.x;
    if (parent_idx >= cell_count || division_flags[parent_idx] == 0u) { return; }
    
    // Get allocation index from prefix-sum
    let division_offset = prefix_sum[parent_idx];
    
    // Child A overwrites parent
    let child_a_idx = parent_idx;
    
    // Child B gets new index
    let child_b_idx = atomicAdd(&cell_count, 1u);
    
    // Get mode data
    let parent_mode = get_mode(genome_index[parent_idx], mode_index[parent_idx]);
    
    // Calculate masses
    let parent_mass = masses[parent_idx];
    let child_a_mass = parent_mass * parent_mode.split_ratio;
    let child_b_mass = parent_mass * (1.0 - parent_mode.split_ratio);
    
    // Calculate positions
    let split_dir = calculate_split_direction(parent_mode.split_angle);
    let split_dir_world = rotate_vector(rotations[parent_idx], split_dir);
    let offset = split_dir_world * radii[parent_idx] * 0.5;
    
    let child_a_pos = positions[parent_idx] + offset;
    let child_b_pos = positions[parent_idx] - offset;
    
    // Calculate orientations
    let child_a_rot = calculate_child_rotation(rotations[parent_idx], parent_mode.child_a_angle);
    let child_b_rot = calculate_child_rotation(rotations[parent_idx], parent_mode.child_b_angle);
    
    // Write child A (overwrites parent)
    positions[child_a_idx] = child_a_pos;
    velocities[child_a_idx] = velocities[parent_idx]; // Inherit velocity
    rotations[child_a_idx] = child_a_rot;
    angular_velocities[child_a_idx] = angular_velocities[parent_idx];
    masses[child_a_idx] = child_a_mass;
    ages[child_a_idx] = 0.0;
    splits[child_a_idx] = 0u;
    genome_index[child_a_idx] = genome_index[parent_idx];
    mode_index[child_a_idx] = parent_mode.child_a_mode;
    // ... (other fields)
    
    // Write child B (new cell)
    positions[child_b_idx] = child_b_pos;
    // ... (similar to child A with child B parameters)
}
```

- [ ] Implement division execution shader
- [ ] Mass splitting
- [ ] Position/orientation calculation
- [ ] Child A overwrites parent
- [ ] Child B allocated via atomic counter
- [ ] Initialize all child properties

#### Adhesion Inheritance

- [ ] Separate shader after division execution
- [ ] For each divided cell, classify parent adhesions
- [ ] Create new adhesions for children
- [ ] Update neighbor adhesions to point to children
- [ ] Complex logic - may need multiple passes

```wgsl
@compute @workgroup_size(256)
fn adhesion_inheritance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // For each parent cell that divided
    // Classify its adhesions into zones A, B, C
    // Create new adhesions for children
    // This is complex on GPU - may be bottleneck
}
```

- [ ] Implement adhesion inheritance shader
- [ ] Zone classification (A, B, C)
- [ ] Geometric anchor recalculation
- [ ] Atomic adhesion allocation
- [ ] Update neighbor adhesions

### Energy and Biomass (GPU)

#### Mass Accumulation

```wgsl
@compute @workgroup_size(256)
fn mass_accumulation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || !active[idx]) { return; }
    
    let cell_type = cell_types[idx];
    
    // Test cells: passive growth
    if (cell_type == CELL_TYPE_TEST) {
        let mode = get_mode(genome_index[idx], mode_index[idx]);
        masses[idx] += mode.nutrient_gain_rate * dt;
        masses[idx] = min(masses[idx], 2.0 * mode.split_mass);
    }
    
    // Photocyte: light-based energy
    if (cell_type == CELL_TYPE_PHOTOCYTE) {
        let grid_pos = position_to_grid(positions[idx]);
        let light = light_grid[grid_pos];
        masses[idx] += light * photocyte_efficiency * dt;
    }
    
    // Chemocyte: dissolved nutrients
    if (cell_type == CELL_TYPE_CHEMOCYTE) {
        let grid_pos = position_to_grid(positions[idx]);
        let nutrient = nutrient_grid[grid_pos];
        let intake = nutrient * chemocyte_rate * dt;
        masses[idx] += intake;
        atomicAdd(&nutrient_grid[grid_pos], -intake); // Deplete nutrients
    }
    
    // Flagellocyte: consumption
    if (cell_type == CELL_TYPE_FLAGELLOCYTE) {
        let signal = signals[idx]; // Propulsion signal
        let force_magnitude = abs(signal) * max_flagellocyte_force;
        masses[idx] -= 0.2 * force_magnitude * dt;
    }
}
```

- [ ] Implement mass accumulation shader
- [ ] Test cell passive growth
- [ ] Photocyte light absorption
- [ ] Chemocyte nutrient absorption
- [ ] Flagellocyte consumption
- [ ] Other cell type mass changes

#### Mass Transfer

```wgsl
@compute @workgroup_size(256)
fn mass_transfer(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= adhesion_count || !adhesion_active[idx]) { return; }
    
    let cell_a = adhesion_cell_a[idx];
    let cell_b = adhesion_cell_b[idx];
    
    // Get priorities (with emergency boost)
    let priority_a = get_priority(cell_a);
    let priority_b = get_priority(cell_b);
    
    // Calculate pressures
    let pressure_a = masses[cell_a] / priority_a;
    let pressure_b = masses[cell_b] / priority_b;
    
    // Calculate flow
    let flow = (pressure_a - pressure_b) * transport_rate * dt;
    
    // Check minimum thresholds
    let min_a = get_minimum_mass(cell_a);
    let min_b = get_minimum_mass(cell_b);
    
    // Transfer mass (atomic operations)
    if (flow > 0.0 && masses[cell_a] - flow >= min_a) {
        atomicAdd(&masses[cell_a], -flow);
        atomicAdd(&masses[cell_b], flow);
    } else if (flow < 0.0 && masses[cell_b] + flow >= min_b) {
        atomicAdd(&masses[cell_a], -flow);
        atomicAdd(&masses[cell_b], flow);
    }
}
```

- [ ] Implement mass transfer shader
- [ ] Pressure-based flow calculation
- [ ] Priority system with emergency boost
- [ ] Minimum mass thresholds
- [ ] Atomic mass updates

#### Cell Death

```wgsl
@compute @workgroup_size(256)
fn cell_death(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || !active[idx]) { return; }
    
    // Check starvation
    if (masses[idx] < 0.5) {
        active[idx] = 0u;
        // Mark adhesions for removal (in separate pass)
        // Release nutrients to grid (if grid available)
    }
}
```

- [ ] Implement cell death shader
- [ ] Starvation check
- [ ] Mark cells inactive
- [ ] Queue adhesion removal
- [ ] Release nutrients to grid

---

## Fluid Simulation

### Navier-Stokes Equations (Simplified)

#### Fluid State

- [ ] Velocity field (3D vector field on 64³ grid)
- [ ] Pressure field (scalar field on 64³ grid)
- [ ] Incompressibility constraint (divergence-free velocity)

#### Simulation Steps

1. [ ] Advection (velocity advects itself)
2. [ ] External forces (cells, gravity, etc.)
3. [ ] Diffusion (viscosity)
4. [ ] Pressure solve (enforce incompressibility)
5. [ ] Pressure gradient (correct velocity)

### Advection Shader

```wgsl
@compute @workgroup_size(8, 8, 8)
fn fluid_advection(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.xyz;
    if (any(grid_idx >= grid_dimensions)) { return; }
    
    // Semi-Lagrangian advection
    let pos = grid_to_position(grid_idx);
    let velocity = velocity_grid[flatten(grid_idx)];
    
    // Trace particle backward in time
    let prev_pos = pos - velocity * dt;
    
    // Interpolate velocity at previous position
    let advected_velocity = interpolate_velocity(prev_pos);
    
    // Write to temporary buffer
    velocity_temp[flatten(grid_idx)] = advected_velocity;
}
```

- [ ] Implement advection shader
- [ ] Semi-Lagrangian method (unconditionally stable)
- [ ] Trilinear interpolation for velocity
- [ ] Handle boundary conditions
- [ ] Write to temporary buffer (double buffering)

### Divergence Shader

```wgsl
@compute @workgroup_size(8, 8, 8)
fn fluid_divergence(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.xyz;
    if (any(grid_idx >= grid_dimensions)) { return; }
    
    // Central differences for divergence
    let u_right = velocity_grid[flatten(grid_idx + vec3(1, 0, 0))].x;
    let u_left = velocity_grid[flatten(grid_idx - vec3(1, 0, 0))].x;
    let v_up = velocity_grid[flatten(grid_idx + vec3(0, 1, 0))].y;
    let v_down = velocity_grid[flatten(grid_idx - vec3(0, 1, 0))].y;
    let w_forward = velocity_grid[flatten(grid_idx + vec3(0, 0, 1))].z;
    let w_backward = velocity_grid[flatten(grid_idx - vec3(0, 0, 1))].z;
    
    let divergence = ((u_right - u_left) + (v_up - v_down) + (w_forward - w_backward)) / (2.0 * cell_size);
    
    divergence_grid[flatten(grid_idx)] = divergence;
}
```

- [ ] Implement divergence shader
- [ ] Central differences (finite differences)
- [ ] Handle boundary conditions (solid walls)
- [ ] Store divergence for pressure solve

### Pressure Solve (Jacobi Iteration)

```wgsl
@compute @workgroup_size(8, 8, 8)
fn fluid_pressure_jacobi(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.xyz;
    if (any(grid_idx >= grid_dimensions)) { return; }
    
    // Jacobi iteration: solve Poisson equation
    // ∇²p = -divergence
    
    let p_left = pressure_grid[flatten(grid_idx - vec3(1, 0, 0))];
    let p_right = pressure_grid[flatten(grid_idx + vec3(1, 0, 0))];
    let p_down = pressure_grid[flatten(grid_idx - vec3(0, 1, 0))];
    let p_up = pressure_grid[flatten(grid_idx + vec3(0, 1, 0))];
    let p_backward = pressure_grid[flatten(grid_idx - vec3(0, 0, 1))];
    let p_forward = pressure_grid[flatten(grid_idx + vec3(0, 0, 1))];
    
    let divergence = divergence_grid[flatten(grid_idx)];
    
    let p_new = (p_left + p_right + p_down + p_up + p_backward + p_forward - divergence * cell_size * cell_size) / 6.0;
    
    pressure_temp[flatten(grid_idx)] = p_new;
}
```

- [ ] Implement Jacobi iteration shader
- [ ] Iterate 20-50 times per tick (dispatch shader multiple times)
- [ ] Or use conjugate gradient solver (more complex, faster)
- [ ] Swap pressure and pressure_temp buffers after each iteration
- [ ] Handle boundary conditions

### Pressure Gradient Shader

```wgsl
@compute @workgroup_size(8, 8, 8)
fn fluid_pressure_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.xyz;
    if (any(grid_idx >= grid_dimensions)) { return; }
    
    // Calculate pressure gradient
    let p_right = pressure_grid[flatten(grid_idx + vec3(1, 0, 0))];
    let p_left = pressure_grid[flatten(grid_idx - vec3(1, 0, 0))];
    let p_up = pressure_grid[flatten(grid_idx + vec3(0, 1, 0))];
    let p_down = pressure_grid[flatten(grid_idx - vec3(0, 1, 0))];
    let p_forward = pressure_grid[flatten(grid_idx + vec3(0, 0, 1))];
    let p_backward = pressure_grid[flatten(grid_idx - vec3(0, 0, 1))];
    
    let gradient = vec3(
        (p_right - p_left) / (2.0 * cell_size),
        (p_up - p_down) / (2.0 * cell_size),
        (p_forward - p_backward) / (2.0 * cell_size)
    );
    
    // Subtract gradient from velocity (make incompressible)
    velocity_grid[flatten(grid_idx)] -= gradient;
}
```

- [ ] Implement pressure gradient shader
- [ ] Calculate gradient using central differences
- [ ] Subtract gradient from velocity
- [ ] Result: divergence-free velocity field

### Cell-Fluid Coupling

#### Cells Affect Fluid

```wgsl
@compute @workgroup_size(256)
fn fluid_cell_coupling(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || !active[idx]) { return; }
    
    // Get grid cell containing this cell
    let grid_pos = position_to_grid(positions[idx]);
    
    // Cells act as obstacles (block flow)
    // Mark grid cell as occupied
    // Reduce velocity in this grid cell
    
    let cell_velocity = velocities[idx];
    let grid_velocity = velocity_grid[flatten(grid_pos)];
    
    // Blend velocities (cell affects fluid)
    let influence = 0.5; // Strength of coupling
    let new_velocity = mix(grid_velocity, cell_velocity, influence);
    
    atomicStore(&velocity_grid[flatten(grid_pos)], new_velocity);
    
    // Flagellocytes create perturbations
    if (cell_types[idx] == CELL_TYPE_FLAGELLOCYTE) {
        let propulsion = get_flagellocyte_force(idx);
        atomicAdd(&velocity_grid[flatten(grid_pos)], propulsion * dt);
    }
}
```

- [ ] Implement cell-to-fluid coupling
- [ ] Cells block/redirect flow
- [ ] Flagellocytes create currents
- [ ] Atomic operations for grid updates

#### Fluid Affects Cells

```wgsl
@compute @workgroup_size(256)
fn fluid_to_cell_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || !active[idx]) { return; }
    
    // Get fluid velocity at cell position
    let grid_pos = position_to_grid(positions[idx]);
    let fluid_velocity = velocity_grid[flatten(grid_pos)];
    
    // Calculate drag force
    let relative_velocity = fluid_velocity - velocities[idx];
    let drag_force = drag_coefficient * relative_velocity;
    
    // Apply force to cell
    atomicAdd(&forces[idx], drag_force);
}
```

- [ ] Implement fluid-to-cell forces
- [ ] Sample fluid velocity at cell position
- [ ] Calculate drag force
- [ ] Apply to cell force accumulator

### Adaptive Update Strategy

- [ ] Mark grid cells as active if:
    - [ ] Contains cells
    - [ ] Has non-zero velocity
    - [ ] Has significant chemical concentration
    - [ ] Neighbors an active cell
- [ ] Only update active cells
- [ ] Reduces computation in empty regions

```wgsl
@compute @workgroup_size(8, 8, 8)
fn update_active_flags(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.xyz;
    // Check criteria and set active flag
}
```

- [ ] Implement active flag update shader
- [ ] Check activation criteria
- [ ] Propagate to neighbors (flood fill)
- [ ] Use active flags in fluid shaders (early exit if inactive)

---

## Environmental Systems

### Chemical Diffusion

```wgsl
@compute @workgroup_size(8, 8, 8)
fn chemical_diffusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.xyz;
    if (any(grid_idx >= grid_dimensions)) { return; }
    
    // For each of 4 chemical types
    for (var chem = 0u; chem < 4u; chem++) {
        let c = chemical_grids[chem][flatten(grid_idx)];
        
        // Get neighbor concentrations
        let c_left = chemical_grids[chem][flatten(grid_idx - vec3(1, 0, 0))];
        let c_right = chemical_grids[chem][flatten(grid_idx + vec3(1, 0, 0))];
        // ... (other neighbors)
        
        // Diffusion equation (finite differences)
        let laplacian = (c_left + c_right + c_down + c_up + c_backward + c_forward - 6.0 * c) / (cell_size * cell_size);
        let c_new = c + diffusion_rate * laplacian * dt;
        
        // Apply decay
        c_new *= exp(-decay_rate * dt);
        
        // Clamp to saturation limit
        c_new = min(c_new, saturation_limit);
        
        chemical_temp[chem][flatten(grid_idx)] = c_new;
    }
}
```

- [ ] Implement chemical diffusion shader
- [ ] Diffusion using Laplacian (finite differences)
- [ ] Exponential decay
- [ ] Saturation clamping
- [ ] Separate diffusion for each of 4 chemicals

### Chemical Advection

```wgsl
@compute @workgroup_size(8, 8, 8)
fn chemical_advection(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.xyz;
    if (any(grid_idx >= grid_dimensions)) { return; }
    
    let pos = grid_to_position(grid_idx);
    let velocity = velocity_grid[flatten(grid_idx)];
    
    // Trace backward
    let prev_pos = pos - velocity * dt;
    
    // For each chemical
    for (var chem = 0u; chem < 4u; chem++) {
        // Interpolate concentration
        let c = interpolate_chemical(chem, prev_pos);
        chemical_temp[chem][flatten(grid_idx)] = c;
    }
}
```

- [ ] Implement chemical advection shader
- [ ] Semi-Lagrangian method
- [ ] Interpolate chemical concentrations
- [ ] Advect with fluid velocity

### Secrocyte Secretion

```wgsl
@compute @workgroup_size(256)
fn secrocyte_secretion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || cell_types[idx] != CELL_TYPE_SECROCYTE) { return; }
    
    // Get signal input (determines secretion rate)
    let signal = signals[idx];
    let secretion_rate = abs(signal) * max_secretion_rate;
    
    // Get grid cell
    let grid_pos = position_to_grid(positions[idx]);
    
    // Get chemical type from mode (TBD)
    let chemical_type = get_secrocyte_chemical_type(idx);
    
    // Add chemical to grid
    let amount = secretion_rate * dt;
    atomicAdd(&chemical_grids[chemical_type][flatten(grid_pos)], amount);
    
    // Consume mass
    masses[idx] -= amount * secretion_cost;
}
```

- [ ] Implement Secrocyte secretion
- [ ] Signal-controlled secretion rate
- [ ] Add chemicals to grid (atomic)
- [ ] Biomass cost for secretion
- [ ] Check saturation limits

### Light System

#### Light Propagation Shader

```wgsl
@compute @workgroup_size(8, 8, 8)
fn light_propagation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let grid_idx = global_id.xyz;
    if (any(grid_idx >= grid_dimensions)) { return; }
    
    // Raycast from light sources through grid
    // Or diffusion-based light spread
    // Or pre-computed light field
    
    // For each light source
    for (var light = 0u; light < light_count; light++) {
        let light_pos = light_positions[light];
        let light_intensity = light_intensities[light];
        
        // Ray from light to this grid cell
        let grid_pos = grid_to_position(grid_idx);
        let direction = normalize(grid_pos - light_pos);
        let distance = length(grid_pos - light_pos);
        
        // Check occlusion by cells
        let occluded = raycast_cells(light_pos, direction, distance);
        
        if (!occluded) {
            // Inverse square falloff
            let intensity = light_intensity / (distance * distance);
            light_grid[flatten(grid_idx)] += intensity;
        }
    }
}
```

- [ ] Implement light propagation shader
- [ ] Raycast or diffusion method
- [ ] Occlusion by cells
- [ ] Inverse square falloff
- [ ] Multiple light sources

#### Cell Occlusion Check

```wgsl
fn raycast_cells(origin: vec3<f32>, direction: vec3<f32>, max_distance: f32) -> bool {
    // Step through spatial grid along ray
    // Check for cell intersections
    // Use DDA or similar algorithm
    // Return true if any cell intersects
}
```

- [ ] Implement ray-cell intersection
- [ ] Use spatial grid for acceleration
- [ ] Early exit on first intersection
- [ ] Efficient traversal algorithm (DDA, 3D-DDA)

### Dissolved Nutrients

```wgsl
@compute @workgroup_size(8, 8, 8)
fn nutrient_diffusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Similar to chemical diffusion
    // Diffuse dissolved nutrients
    // Chemocytes consume, Nitrocytes produce
}
```

- [ ] Implement nutrient diffusion (similar to chemicals)
- [ ] Consumption by Chemocytes (atomic subtract)
- [ ] Production by Nitrocytes (atomic add)
- [ ] Cell death releases nutrients

### Gas System

```wgsl
@compute @workgroup_size(8, 8, 8)
fn gas_diffusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // O2, CO2 diffusion
    // Photocytes produce O2
    // Cells consume O2 (respiration)
}
```

- [ ] Implement gas diffusion
- [ ] O2 production (Photocytes)
- [ ] O2 consumption (all cells)
- [ ] CO2 production/consumption
- [ ] Gas exchange at world boundary (optional)

---

## Cell Types Implementation

### Sensor Cells

#### Oculocyte (Forward Vision)

```wgsl
@compute @workgroup_size(256)
fn oculocyte_sense(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || cell_types[idx] != CELL_TYPE_OCULOCYTE) { return; }
    
    // Get forward direction from rotation
    let forward = rotate_vector(rotations[idx], vec3(0.0, 0.0, 1.0));
    
    // Define cone parameters from mode
    let cone_angle = get_oculocyte_cone_angle(idx);
    let max_range = get_oculocyte_range(idx);
    let target_type = get_oculocyte_target(idx);
    
    // Raycast in forward direction
    // Check grid cells in cone
    // Find nearest target of specified type
    
    let nearest_distance = max_range;
    // ... (spatial query for targets)
    
    // Generate signal proportional to distance
    let signal = 1.0 - (nearest_distance / max_range);
    signals_out[idx] = signal;
}
```

- [ ] Implement Oculocyte sensing
- [ ] Forward direction from rotation
- [ ] Cone-based spatial query
- [ ] Grid cell iteration in cone
- [ ] Distance-based signal generation
- [ ] Target type filtering

#### Senseocyte (Omnidirectional)

- [ ] Similar to Oculocyte but omnidirectional
- [ ] Query sphere around cell
- [ ] Find nearest target of any direction

#### Stereocyte (Position Sensing)

- [ ] Find target in range
- [ ] Calculate relative position in local coords
- [ ] Output 2D or 3D position signal

#### Velocity Sensor

- [ ] Track target from previous tick
- [ ] Calculate velocity change
- [ ] Output relative velocity in local coords

### Active Cells

#### Flagellocyte (Propulsion)

```wgsl
@compute @workgroup_size(256)
fn flagellocyte_propulsion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || cell_types[idx] != CELL_TYPE_FLAGELLOCYTE) { return; }
    
    // Read signal input
    let signal = signals_in[idx];
    
    // Get forward direction
    let forward = rotate_vector(rotations[idx], vec3(0.0, 0.0, 1.0));
    
    // Calculate propulsion force
    let force_magnitude = signal * max_flagellocyte_force;
    let propulsion_force = forward * force_magnitude;
    
    // Apply force
    atomicAdd(&forces[idx], propulsion_force);
    
    // Consume mass
    masses[idx] -= 0.2 * abs(force_magnitude) * dt;
    
    // Create current in fluid grid
    let grid_pos = position_to_grid(positions[idx]);
    atomicAdd(&velocity_grid[flatten(grid_pos)], propulsion_force * fluid_coupling_strength);
}
```

- [ ] Implement Flagellocyte propulsion
- [ ] Signal-controlled force
- [ ] Forward direction from rotation
- [ ] Mass consumption
- [ ] Fluid perturbation

#### Myocyte (Muscle Actuation)

- [ ] Read multiple signal inputs (bend1, bend2, contract, twist)
- [ ] Modify adhesion properties (rest length, angles)
- [ ] Apply forces/torques to adhered cells
- [ ] Mass consumption

#### Neurocyte (Signal Processing)

```wgsl
@compute @workgroup_size(256)
fn neurocyte_process(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || cell_types[idx] != CELL_TYPE_NEUROCYTE) { return; }
    
    // Read input signals (4 signals)
    let inputs = signals_in_array[idx]; // vec4<f32>
    
    // Get neural function from mode (logic gates, math, etc.)
    let function_type = get_neurocyte_function(idx);
    
    // Compute output based on function
    var output = 0.0;
    switch (function_type) {
        case NEUROCYTE_AND: { output = min(inputs.x, inputs.y); }
        case NEUROCYTE_OR: { output = max(inputs.x, inputs.y); }
        case NEUROCYTE_NOT: { output = 1.0 - inputs.x; }
        case NEUROCYTE_ADD: { output = inputs.x + inputs.y; }
        case NEUROCYTE_MULTIPLY: { output = inputs.x * inputs.y; }
        // ... (other functions)
    }
    
    // Clamp output to [-1, 1]
    output = clamp(output, -1.0, 1.0);
    
    signals_out[idx] = output;
}
```

- [ ] Implement Neurocyte signal processing
- [ ] Multiple function types (logic, math)
- [ ] Input signal reading
- [ ] Output signal writing
- [ ] Function defined in mode

### Passive Cells (Grid-Dependent)

#### Phagocyte (Food Consumption)

```wgsl
@compute @workgroup_size(256)
fn phagocyte_eat(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || cell_types[idx] != CELL_TYPE_PHAGOCYTE) { return; }
    
    // Check for food pellets in collision range
    // Use spatial grid to find nearby pellets
    
    let nearest_pellet = find_nearest_pellet(idx);
    
    if (nearest_pellet != INVALID) {
        // Consume pellet
        let pellet_mass = pellet_masses[nearest_pellet];
        masses[idx] += pellet_mass * consumption_efficiency;
        
        // Mark pellet as consumed
        pellet_active[nearest_pellet] = 0u;
    }
}
```

- [ ] Implement Phagocyte food consumption
- [ ] Spatial query for food pellets
- [ ] Collision/proximity check
- [ ] Mass transfer from pellet
- [ ] Mark pellet inactive

#### Chemocyte (Already Implemented)

- [ ] Grid-based nutrient absorption
- [ ] Covered in mass accumulation section

#### Photocyte (Already Implemented)

- [ ] Grid-based light absorption
- [ ] Covered in mass accumulation section

#### Nitrocyte (Nutrient Production)

```wgsl
@compute @workgroup_size(256)
fn nitrocyte_produce(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || cell_types[idx] != CELL_TYPE_NITROCYTE) { return; }
    
    // Produce nutrients to grid
    let grid_pos = position_to_grid(positions[idx]);
    let production_rate = nitrocyte_rate * dt;
    
    atomicAdd(&nutrient_grid[flatten(grid_pos)], production_rate);
    
    // Consume mass
    masses[idx] -= production_rate * production_cost;
}
```

- [ ] Implement Nitrocyte nutrient production
- [ ] Add nutrients to grid (atomic)
- [ ] Mass consumption

#### Glueocyte (Contact Adhesion)

```wgsl
@compute @workgroup_size(256)
fn glueocyte_contact(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || cell_types[idx] != CELL_TYPE_GLUEOCYTE) { return; }
    
    // Check for collisions with other cells
    // Use collision detection results
    
    for (var i = 0u; i < collision_count[idx]; i++) {
        let other_cell = collision_partners[idx * MAX_COLLISIONS + i];
        
        // Check if adhesion already exists
        if (!has_adhesion(idx, other_cell)) {
            // Create new adhesion (queue for creation)
            queue_adhesion_creation(idx, other_cell);
        }
    }
}
```

- [ ] Implement Glueocyte contact adhesions
- [ ] Use collision detection results
- [ ] Check existing adhesions (expensive on GPU)
- [ ] Queue adhesion creation
- [ ] Respect adhesion limits

#### Devorocyte (Mass Drain)

```wgsl
@compute @workgroup_size(256)
fn devorocyte_drain(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || cell_types[idx] != CELL_TYPE_DEVOROCYTE) { return; }
    
    // Check for contact with other cells
    for (var i = 0u; i < collision_count[idx]; i++) {
        let victim = collision_partners[idx * MAX_COLLISIONS + i];
        
        // Check if victim is protected (Keratinocyte)
        if (cell_types[victim] != CELL_TYPE_KERATINOCYTE) {
            // Drain mass
            let drain_rate = devorocyte_rate * dt;
            let drained = min(drain_rate, masses[victim] - 0.5);
            
            atomicAdd(&masses[victim], -drained);
            atomicAdd(&masses[idx], drained);
        }
    }
}
```

- [ ] Implement Devorocyte mass drain
- [ ] Contact detection
- [ ] Keratinocyte protection check
- [ ] Atomic mass transfer

---

## GPU-Driven Rendering

### Indirect Rendering

#### Compute Culling and LOD

```wgsl
@compute @workgroup_size(256)
fn rendering_prepare(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cell_count || !active[idx]) { return; }
    
    // Frustum culling
    if (!in_frustum(positions[idx], radii[idx])) {
        return; // Skip this cell
    }
    
    // Calculate LOD level based on distance
    let distance = length(positions[idx] - camera_position);
    let lod = calculate_lod(distance);
    
    // Write to instance buffer (compacted)
    let instance_idx = atomicAdd(&instance_count, 1u);
    instance_positions[instance_idx] = positions[idx];
    instance_rotations[instance_idx] = rotations[idx];
    instance_scales[instance_idx] = radii[idx];
    instance_colors[instance_idx] = get_color(idx);
    instance_lod[instance_idx] = lod;
}
```

- [ ] Implement rendering preparation shader
- [ ] Frustum culling on GPU
- [ ] LOD calculation
- [ ] Compact instance data (only visible cells)
- [ ] Write to instance buffers

#### Indirect Draw Commands

```wgsl
struct IndirectDrawCommand {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

@compute @workgroup_size(1)
fn generate_draw_commands() {
    // Generate draw commands for each LOD level
    // Based on instance_count from culling shader
    
    for (var lod = 0u; lod < 4u; lod++) {
        draw_commands[lod].vertex_count = lod_vertex_counts[lod];
        draw_commands[lod].instance_count = lod_instance_counts[lod];
        draw_commands[lod].first_vertex = 0u;
        draw_commands[lod].first_instance = lod_instance_offsets[lod];
    }
}
```

- [ ] Generate indirect draw commands on GPU
- [ ] One draw call per LOD level
- [ ] No CPU readback required
- [ ] Maximum rendering performance

### Rendering Pipeline

- [ ] Vertex shader reads instance data
- [ ] Apply transformations (position, rotation, scale)
- [ ] Fragment shader applies lighting and color
- [ ] Render all cells in minimal draw calls

---

## Performance Optimization

### GPU-Specific Optimizations

#### Memory Coalescing

- [ ] Ensure sequential threads access sequential memory
- [ ] SoA layout helps with coalescing
- [ ] Avoid scattered memory access patterns

#### Occupancy Optimization

- [ ] Balance register usage vs thread count
- [ ] More threads = better occupancy, but limited registers
- [ ] Profile and tune workgroup size
- [ ] Use shared memory to reduce register pressure

#### Shared Memory Usage

- [ ] Use workgroup shared memory for frequently accessed data
- [ ] Prefix-sum uses shared memory extensively
- [ ] Avoid bank conflicts (sequential access by sequential threads)

#### Warp Divergence Avoidance

- [ ] Minimize branching (if statements)
- [ ] Ensure threads in a warp take same code path
- [ ] Use select() instead of if/else when possible
- [ ] Predicates can be faster than branches

#### Atomic Operation Minimization

- [ ] Atomics are slow (serialize threads)
- [ ] Use local accumulation then single atomic when possible
- [ ] Consider lock-free algorithms
- [ ] Per-workgroup reduction before global atomic

### Compute Shader Optimization

#### Minimize Dispatches

- [ ] Combine multiple operations into single shader if possible
- [ ] Balance: fewer dispatches vs shader complexity
- [ ] Each dispatch has overhead

#### Pipeline Barriers

- [ ] Only barrier when necessary (data dependency)
- [ ] Buffer barriers faster than full pipeline barriers
- [ ] Group barriers to minimize stalls

#### Buffer Layout

- [ ] 16-byte alignment for best performance
- [ ] Pad structs to avoid misalignment
- [ ] Use vec4 instead of vec3 for alignment

### Profiling

#### GPU Profiling Tools

- [ ] Use RenderDoc, NSight, or similar
- [ ] Profile shader execution time
- [ ] Identify bottlenecks (memory, compute, barriers)
- [ ] Measure occupancy

#### Performance Metrics

- [ ] Ticks per second (target: >50 TPS)
- [ ] Time per shader (microseconds)
- [ ] Memory bandwidth usage
- [ ] Compute utilization
- [ ] Frieza units per cell (target: <0.2Fr)

### Scalability

#### Variable Cell Count

- [ ] Ensure performance scales with cell count
- [ ] Test at 10K, 50K, 100K, 150K, 200K cells
- [ ] Identify where performance degrades
- [ ] Optimize critical shaders

#### GPU Hardware Tiers

- [ ] Test on low-end, mid-range, high-end GPUs
- [ ] Quality/performance settings
- [ ] Disable expensive features on low-end (fluid, chemicals, LOD)

---

## Testing & Validation

### Unit Tests (CPU-Side)

#### Buffer Management Tests

- [ ] Buffer creation and initialization
- [ ] Binding group setup
- [ ] Buffer uploads and downloads
- [ ] Alignment verification

#### Shader Compilation Tests

- [ ] All shaders compile without errors
- [ ] Shader validation passes
- [ ] Binding group layouts match shader expectations

### Integration Tests (GPU Execution)

#### Physics Tests

- [ ] Run physics shaders on GPU
- [ ] Download results to CPU
- [ ] Verify correctness against CPU reference
- [ ] Energy conservation (approximate)
- [ ] Momentum conservation (approximate)

#### Simulation Tests

- [ ] Run 1000 ticks on GPU
- [ ] No crashes or hangs
- [ ] Cell count matches expected
- [ ] Organism shapes reasonable
- [ ] No NaN or Inf values in buffers

### Parity Tests

#### GPU vs CPU Parity

- [ ] Run same genome on GPU and CPU
- [ ] Compare cell counts over time
- [ ] Compare organism shapes (approximate)
- [ ] Acceptable divergence (<10% after 1000 ticks)

#### GPU vs Preview Parity

- [ ] Run same genome on GPU and Preview
- [ ] Similar results expected (not identical)
- [ ] Organism develops similarly

### Performance Tests

- [ ] Measure TPS at various cell counts
- [ ] Verify <0.2Fr at 100K cells
- [ ] Profile shader execution times
- [ ] Identify bottlenecks
- [ ] Optimize critical shaders
- [ ] Achieve target performance

### Stress Tests

- [ ] 200K cells running for hours
- [ ] 200K cells all dividing simultaneously
- [ ] Maximum adhesions (200K cells × 10 = 2M adhesions)
- [ ] Fluid simulation at max resolution
- [ ] All environmental systems active
- [ ] No crashes, no performance degradation

### Visual Validation

- [ ] Render organisms from GPU simulation
- [ ] Visual appearance matches expectations
- [ ] Smooth animation (60 FPS)
- [ ] No visual artifacts (flickering, popping)
- [ ] LOD transitions smooth

### Determinism Tests

- [ ] Run same genome multiple times
- [ ] Results similar (not identical due to GPU variance)
- [ ] Long-term stability consistent
- [ ] No divergence catastrophes

---

## Implementation Phases

### Phase 1: Foundation

- [ ] Set up GPU module structure
- [ ] Initialize WGPU/Bevy compute pipeline
- [ ] Create basic buffer management
- [ ] Implement simple compute shader (test)
- [ ] Verify GPU execution

### Phase 2: Basic Physics on GPU

- [ ] Implement clear forces shader
- [ ] Implement integration shader
- [ ] Test with a few hundred cells (CPU upload/download)
- [ ] Verify correctness against CPU

### Phase 3: Spatial Grid on GPU

- [ ] Implement grid rebuild shader
- [ ] Fixed-size grid cell approach (simple)
- [ ] Test grid rebuild correctness

### Phase 4: Collisions on GPU

- [ ] Implement collision detection shader
- [ ] Implement collision response shader
- [ ] Integrate with force accumulation
- [ ] Test with collision scenarios

### Phase 5: Adhesions on GPU

- [ ] Implement adhesion force shader
- [ ] Upload adhesion data to GPU
- [ ] Test with pre-created adhesions
- [ ] Verify forces match CPU

### Phase 6: Cell Division on GPU

- [ ] Implement division detection shader
- [ ] Implement prefix-sum for allocation
- [ ] Implement division execution shader
- [ ] Test simple divisions
- [ ] Adhesion inheritance (complex, may defer)

### Phase 7: Genome Integration

- [ ] Upload genome library to GPU
- [ ] Reference genomes from cells
- [ ] Mode data access in shaders
- [ ] Test with multiple genomes

### Phase 8: Energy System on GPU

- [ ] Mass accumulation shaders
- [ ] Mass transfer shader
- [ ] Cell death shader
- [ ] Test energy conservation

### Phase 9: Fluid Simulation

- [ ] Implement advection shader
- [ ] Implement divergence shader
- [ ] Implement pressure solve (Jacobi)
- [ ] Implement pressure gradient shader
- [ ] Test fluid flow

### Phase 10: Cell-Fluid Coupling

- [ ] Cells affect fluid shader
- [ ] Fluid affects cells shader
- [ ] Test coupling behavior

### Phase 11: Environmental Systems

- [ ] Chemical diffusion/advection shaders
- [ ] Light propagation shader
- [ ] Nutrient diffusion shader
- [ ] Test environmental interactions

### Phase 12: Cell Types

- [ ] Implement sensor cells (Oculocyte, etc.)
- [ ] Implement active cells (Flagellocyte, etc.)
- [ ] Implement passive cells (Phagocyte, etc.)
- [ ] Test all cell types

### Phase 13: GPU-Driven Rendering

- [ ] Implement culling shader
- [ ] Implement LOD selection
- [ ] Generate instance data on GPU
- [ ] Indirect draw commands
- [ ] Test rendering performance

### Phase 14: Performance Optimization

- [ ] Profile all shaders
- [ ] Optimize bottlenecks
- [ ] Tune workgroup sizes
- [ ] Achieve <0.2Fr target
- [ ] Test on various GPUs

### Phase 15: Polish and Testing

- [ ] Comprehensive testing (all test categories)
- [ ] Bug fixes
- [ ] Parity testing with CPU/Preview
- [ ] Visual validation
- [ ] Stress testing

---

## Known Challenges (GPU-Specific)

### Variable-Length Data Structures

- **Problem:** Grid cells have variable number of cells
- **GPU Challenge:** No dynamic arrays, fixed memory allocation
- **Solutions:** Fixed-size arrays (wasteful) or compaction (complex)

### Atomic Operations

- **Problem:** Multiple threads write to same cell/grid
- **Performance:** Atomics serialize threads, slow down
- **Mitigation:** Minimize atomics, use local accumulation

### Debugging

- **Problem:** Hard to debug GPU shaders
- **Tools:** RenderDoc, printf debugging (limited)
- **Strategy:** Validate against CPU, extensive testing

### Adhesion Inheritance

- **Problem:** Complex logic with variable topology
- **GPU Challenge:** Irregular memory access, branching
- **May Defer:** Implement on CPU initially, move to GPU later

### Synchronization

- **Problem:** No cross-workgroup sync in WGSL
- **Solution:** Multiple shader dispatches with barriers
- **Trade-off:** More dispatches = more overhead

---

## Success Criteria

### Functional Requirements Met

- [ ] 100K-200K cells simulated on GPU
- [ ] All physics systems implemented (collisions, adhesions, division)
- [ ] Fluid simulation (64³ grid) running
- [ ] Environmental systems (chemicals, light, nutrients)
- [ ] All cell types implemented
- [ ] GPU-driven rendering with LOD
- [ ] Similar results to CPU/Preview modes

### Performance Requirements Met

- [ ] < 0.2Fr (100K cells at >50 TPS)
- [ ] <20ms per tick at 100K cells
- [ ] Smooth rendering at 60 FPS
- [ ] Efficient GPU utilization (>80%)
- [ ] Scales to 200K cells
- [ ] No performance degradation over time

### Quality Requirements Met

- [ ] Approximate determinism (similar results across runs)
- [ ] Energy conservation (approximate, no explosions)
- [ ] Momentum conservation (approximate)
- [ ] Stable long-term simulation (hours without issues)
- [ ] Accurate organism development
- [ ] Parity with CPU/Preview (<10% divergence)
- [ ] No crashes or GPU hangs

---

## Notes

### Design Decisions

#### Compute-Only Architecture

- All physics runs on GPU (no CPU fallback)
- Rendering driven by GPU (indirect draws)
- Maximizes GPU utilization
- Minimizes CPU-GPU synchronization

#### Buffer Management Strategy

- Triple buffering for async execution
- Pre-allocated fixed-size buffers
- SoA layout for memory coalescing
- 16-byte alignment for performance

#### Fluid Simulation Approach

- Simplified Navier-Stokes (no turbulence)
- Jacobi iteration for pressure (simple, parallelizable)
- Semi-Lagrangian advection (unconditionally stable)
- 64³ grid resolution (balance quality vs performance)

#### Cell Division on GPU

- Prefix-sum for deterministic allocation
- Single-threaded division execution (CPU) initially
- Move to GPU after validation
- Complex adhesion inheritance may stay on CPU

### Common Pitfalls to Avoid

- **Excessive atomics:** Serialize threads, kill performance
- **Memory misalignment:** Cause slowdowns
- **Too many dispatches:** Each has overhead
- **Warp divergence:** Branches hurt performance
- **Uncoalesced memory access:** Waste bandwidth
- **Not profiling:** Guessing is bad, measure first
- **Debugging on GPU:** Very hard, validate against CPU

### Performance Considerations

- **Adhesion forces:** Main workload (100K+ adhesions)
- **Fluid simulation:** Second biggest cost (64³ × iterations)
- **Cell division:** Prefix-sum is expensive
- **Rendering:** GPU-driven is very efficient
- **Atomic contention:** Minimize writes to same memory

### Reference Implementations

- CPU simulation (reference for physics)
- GPU Gems / GPU Pro (compute shader techniques)
- Navier-Stokes fluid solvers (GPU implementations)
- Parallel prefix-sum algorithms (GPU Gems 3 Ch 39)

---

## Changelog

**v1.0** - Initial comprehensive checklist

- All core systems documented
- Complete compute shader pipeline
- Fluid and environmental systems
- All cell types
- GPU-driven rendering
- Performance optimization strategies
- Testing and validation
- Implementation phases
- Known challenges and solutions