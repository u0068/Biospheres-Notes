# Implementation Checklist

## Overview

The CPU simulation is a multi-threaded mode designed for medium-scale organism testing and manual exploration. It uses parallel processing to handle 5K-10K cells efficiently while maintaining similar physics to Preview and GPU modes.

**Core Requirements:**

- **Cell Count:** 5K-10K cells (user configurable)
- **Performance Target:** 10K cells at >50tps = <2Fr
- **Execution:** Multi-threaded with thread pool
- **Control:** Manual time scrubber only, no automatic progression
- **Determinism:** As close as possible to Preview, but not bit-for-bit identical
- **Feature Parity:** Same physics as Preview/GPU, optimized for CPU parallelism

---

## Table of Contents

1. [[#Core Architecture]]
2. [[#Multi-Threading System]]
3. [[#Physics Systems]]
4. [[#Cell Types Implementation]]
5. [[#Genome System Integration]]
6. [[#Rendering System]]
7. [[#UI Integration]]
8. [[#Performance Optimization]]
9. [[#Testing & Validation]]

---

## Core Architecture

### Data Structures

#### Cell Data (Structure of Arrays)

- [ ] Position array (`Vec<Vec3>` with capacity pre-allocated)
- [ ] Velocity array (`Vec<Vec3>`)
- [ ] Rotation array (`Vec<Quat>`)
- [ ] Angular velocity array (`Vec<Vec3>`)
- [ ] Mass array (`Vec<f32>`)
- [ ] Radius array (`Vec<f32>`)
- [ ] Age array (`Vec<f32>`)
- [ ] Split count array (`Vec<u32>`)
- [ ] Genome index array (`Vec<u32>`)
- [ ] Mode index array (`Vec<u8>`)
- [ ] Cell type array (`Vec<u8>`)
- [ ] Active flag array (`Vec<bool>`)
- [ ] Cell count tracker (`AtomicUsize` for thread-safe access)
- [ ] Force accumulator array (`Vec<Vec3>`)
- [ ] Torque accumulator array (`Vec<Vec3>`)

#### Thread-Safe Considerations

- [ ] Read-only data can be shared freely
- [ ] Write operations need synchronization or partitioning
- [ ] Force/torque accumulators need atomic operations or per-thread buffers
- [ ] Cell allocation/deletion requires careful synchronization

#### Adhesion Data

- [ ] Adhesion array (`Vec<Adhesion>` with capacity pre-allocated)
- [ ] Adhesion fields (same as Preview):
    - [ ] Cell A index
    - [ ] Cell B index
    - [ ] Anchor A (local direction vector)
    - [ ] Anchor B (local direction vector)
    - [ ] Rest length
    - [ ] Stiffness
    - [ ] Damping coefficient
    - [ ] Orientation stiffness
    - [ ] Orientation damping
    - [ ] Twist stiffness (optional)
    - [ ] Twist damping (optional)
    - [ ] Twist reference quaternions (optional)
    - [ ] Original side assignment
    - [ ] Active flag
- [ ] Adhesion count tracker (`AtomicUsize`)
- [ ] Thread-safe adhesion creation queue
- [ ] Thread-safe adhesion deletion queue

#### Genome Library

- [ ] Genome deduplication table (`Arc<RwLock<HashMap<GenomeHash, GenomeData>>>`)
- [ ] Thread-safe genome reference counting
- [ ] Active genome indices for loaded organisms
- [ ] Genome data structure (same as Preview)

#### Spatial Grid

- [ ] Grid structure (`Vec<Vec<usize>>` - each cell contains list of cell indices)
- [ ] Grid dimensions and parameters
- [ ] Grid rebuild strategy (parallel or sequential)
- [ ] Thread-safe grid updates or per-thread grids
- [ ] Grid cell size optimization for 10K cells

### Simulation State

- [ ] Current tick (`AtomicU64` for thread-safe access)
- [ ] Timestep size (`f32`, default 0.02)
- [ ] Paused flag (`AtomicBool`)
- [ ] Target tick (for scrubber) (`AtomicU64`)
- [ ] Simulation bounds (same as Preview)
- [ ] Thread pool instance
- [ ] Number of worker threads (configurable, default to CPU core count)

### Memory Management

- [ ] Pre-allocated capacity for 10K cells
- [ ] Pre-allocated capacity for ~50K adhesions (5 per cell average)
- [ ] Dynamic growth strategy if limits exceeded
- [ ] Cell allocation (append to vector, track with indices)
- [ ] Cell deletion (mark inactive, periodic compaction)
- [ ] Adhesion cleanup (periodic compaction)
- [ ] Genome cleanup (remove unreferenced genomes)

---

## Multi-Threading System

### Thread Pool Management

- [ ] Initialize thread pool (rayon or custom)
- [ ] Configurable thread count (default: num_cpus)
- [ ] Thread affinity settings (optional)
- [ ] Thread-local storage for temporary buffers
- [ ] Graceful shutdown on simulation end

### Work Partitioning Strategies

#### Static Partitioning

- [ ] Divide cells into chunks (ceil(cell_count / thread_count))
- [ ] Each thread processes its chunk
- [ ] Simple, predictable, good cache locality
- [ ] May have load imbalance if work varies per cell

#### Dynamic Work Stealing

- [ ] Work queue with tasks
- [ ] Threads steal work from others when idle
- [ ] Better load balancing
- [ ] More overhead, less cache locality

#### Hybrid Approach

- [ ] Static partitioning for main loops
- [ ] Dynamic scheduling for irregular work (division, adhesion creation)
- [ ] Recommended approach

### Synchronization Points

- [ ] Barrier after force calculation (all forces must be computed)
- [ ] Barrier after integration (all positions/velocities updated)
- [ ] Barrier after grid rebuild (grid must be complete)
- [ ] Minimize barriers to reduce thread idle time
- [ ] Use lock-free algorithms where possible

### Data Race Prevention

#### Read-Only Sharing

- [ ] Position, velocity, rotation (read during force calculation)
- [ ] Mass, radius (read-only during physics step)
- [ ] Genome data (read-only during simulation)

#### Write Conflicts

- [ ] Force/torque accumulation (multiple threads writing to same cell)
- [ ] Solutions:
    - [ ] Atomic operations (slower but simple)
    - [ ] Per-thread force buffers, reduce at barrier
    - [ ] Partition cells so no overlap in writes

#### Cell Creation/Deletion

- [ ] Queue operations during parallel phase
- [ ] Process queue in single-threaded phase
- [ ] Or use lock-free concurrent data structures

### Thread-Local Buffers

- [ ] Per-thread force accumulator (reduces contention)
- [ ] Per-thread torque accumulator
- [ ] Per-thread collision pair list
- [ ] Per-thread division queue
- [ ] Reduce/merge buffers at synchronization points

---

## Physics Systems

### Velocity Verlet Integration

#### Parallel Integration Loop

- [ ] Parallel iteration over active cells
- [ ] Calculate forces for cell chunk (thread-safe)
- [ ] Calculate torques for cell chunk (thread-safe)
- [ ] Barrier: wait for all forces/torques computed
- [ ] Update velocities: `v += (force / mass) * dt`
- [ ] Update angular velocities: `ω += (torque / inertia) * dt`
- [ ] Update positions: `pos += velocity * dt`
- [ ] Update rotations: `rot = integrate_rotation(rot, ω, dt)`
- [ ] Clamp velocities and angular velocities
- [ ] Barrier: wait for all integrations complete

#### Force Accumulation Strategy

- [ ] **Option A: Atomic Operations**
    - [ ] Use atomic add for force/torque accumulation
    - [ ] Simple but potentially slower with high contention
- [ ] **Option B: Per-Thread Buffers (Recommended)**
    - [ ] Each thread writes to its own buffer
    - [ ] Reduce phase sums all buffers into final forces
    - [ ] Better performance, more memory usage
- [ ] **Option C: Cell Partitioning**
    - [ ] Partition cells so each thread owns a disjoint set
    - [ ] No write conflicts, best performance
    - [ ] Requires careful partitioning of adhesions

### Collision Detection

#### Parallel Spatial Grid Rebuild

- [ ] **Option A: Sequential Grid Rebuild**
    - [ ] Single-threaded grid clear and rebuild
    - [ ] Simple, no synchronization issues
    - [ ] Potential bottleneck at 10K cells
- [ ] **Option B: Parallel Grid Insert**
    - [ ] Parallel iteration over cells
    - [ ] Each thread inserts cells into grid (requires locks or atomics)
    - [ ] Lock per grid cell or lock-free concurrent data structure
- [ ] **Option C: Per-Thread Grids**
    - [ ] Each thread builds its own grid
    - [ ] Merge grids at end (complex)
    - [ ] Probably not worth the complexity

#### Parallel Collision Pair Generation

- [ ] Partition grid cells among threads
- [ ] Each thread generates collision pairs for its grid cells
- [ ] Store pairs in thread-local lists
- [ ] Concatenate lists or process immediately

#### Parallel Collision Response

- [ ] Iterate over collision pairs in parallel
- [ ] Calculate overlap and forces
- [ ] Accumulate forces to cells (use chosen accumulation strategy)
- [ ] Handle large penetrations (same as Preview)

#### World Boundary Collision

- [ ] Parallel iteration over cells
- [ ] Calculate boundary forces/torques per cell
- [ ] Accumulate to force/torque buffers
- [ ] Independent operations, fully parallel

### Adhesion System

#### Parallel Adhesion Forces

- [ ] Partition adhesions among threads
- [ ] Each thread calculates forces/torques for its adhesions
- [ ] Accumulate forces to cells (use chosen accumulation strategy)
- [ ] Highly parallel, main physics workload

#### Adhesion Creation (Cell Division)

- [ ] Division happens in single-threaded phase (or carefully synchronized)
- [ ] Create adhesions sequentially or queue for creation
- [ ] Geometric anchor calculation (same as Preview)
- [ ] Check adhesion limits

#### Adhesion Breaking

- [ ] Check forces during parallel adhesion calculation
- [ ] Mark adhesions for deletion (don't delete immediately)
- [ ] Process deletion queue in single-threaded phase

#### Adhesion Cleanup

- [ ] Periodic compaction (not every tick)
- [ ] Single-threaded compaction phase
- [ ] Remove inactive adhesions, update indices

### Cell Division

#### Division Detection

- [ ] Parallel iteration checking division criteria
- [ ] Mark cells ready to divide
- [ ] Queue division operations

#### Division Execution

- [ ] **Single-threaded phase** (simpler, recommended for CPU)
- [ ] Process division queue sequentially
- [ ] Perform mass splitting, position calculation
- [ ] Create adhesions with inheritance
- [ ] Update genome references
- [ ] Deterministic within single-threaded context

#### Compaction Phase

- [ ] After all divisions processed
- [ ] Compact cell array if needed (periodic)
- [ ] Update adhesion indices if cells moved

### Energy and Biomass

#### Parallel Mass Updates

- [ ] Passive growth (parallel iteration)
- [ ] Flagellocyte consumption (parallel)
- [ ] Starvation checking (parallel, queue deaths)

#### Mass Transfer Between Cells

- [ ] Iterate over adhesions in parallel
- [ ] Calculate pressure and flow
- [ ] Apply mass changes
- [ ] Potential write conflict: both cells in adhesion
- [ ] Solutions:
    - [ ] Atomic operations on mass
    - [ ] Partition adhesions to avoid conflicts
    - [ ] Sequential mass transfer phase

#### Cell Death

- [ ] Queue deaths during parallel phase
- [ ] Process death queue in single-threaded phase
- [ ] Remove cells and adhesions
- [ ] Update counts

---

## Cell Types Implementation

### Same as Preview, But Parallel

#### Implementation Strategy

- [ ] Each cell type updates in parallel
- [ ] Cell type specific logic is read-only from cell's perspective
- [ ] Writes to cell state use accumulation strategy
- [ ] Grid-dependent types (Phagocyte, Chemocyte, Photocyte) require grid access

### Passive Cell Types

- [ ] Test Cell / Chronocyte
- [ ] Phagocyte (requires food system)
- [ ] Chemocyte (requires chemical grid)
- [ ] Photocyte (requires light grid)
- [ ] Lipocyte
- [ ] Nitrocyte (requires chemical grid)
- [ ] Glueocyte (requires collision detection)
- [ ] Devorocyte (requires collision detection)
- [ ] Keratinocyte

### Active Cell Types

- [ ] Flagellocyte (requires signal system)
- [ ] Myocyte (requires signal system)
- [ ] Secrocyte (requires signal + chemical grid)
- [ ] Stem Cell (requires chemical grid)
- [ ] Neurocyte (requires signal system)
- [ ] Cilliocyte (requires signal + collision detection)
- [ ] Audiocyte (requires signal + audio system)

### Sensor Cell Types

- [ ] Oculocyte (requires spatial queries)
- [ ] Senseocyte (requires spatial queries)
- [ ] Stereocyte (requires spatial queries)
- [ ] Velocity Sensor (requires tracking)

### Thread Safety Considerations

- [ ] Reading from grid (read-only during physics step)
- [ ] Writing to grid (needs synchronization or buffering)
- [ ] Signal propagation (may need separate phase)

---

## Genome System Integration

### Thread-Safe Genome Access

- [ ] Genome library protected by RwLock or similar
- [ ] Read genome data frequently (many threads)
- [ ] Write genome data rarely (genome updates)
- [ ] Use Arc for shared ownership across threads

### Genome Editor Connection

- [ ] Load genome from editor
- [ ] Validate genome structure
- [ ] Add to deduplication table
- [ ] Spawn initial cells with genome

### Genome Hot-Reload

- [ ] Pause simulation
- [ ] Clear simulation state (single-threaded)
- [ ] Load new genome
- [ ] Spawn initial cell(s)
- [ ] Resume simulation

### Initial Cell Placement

- [ ] User clicks to place cell (manual)
- [ ] Default placement (center)
- [ ] Can place multiple cells with different genomes
- [ ] Each cell gets genome reference

---

## Rendering System

### Thread-Safe Rendering Data

#### Double Buffering

- [ ] Physics writes to buffer A
- [ ] Render reads from buffer B
- [ ] Swap buffers at frame boundary (atomic operation)
- [ ] Prevents data races between physics and rendering

#### Instance Data Extraction

- [ ] After physics step completes
- [ ] Copy cell positions, rotations, radii, colors to instance buffer
- [ ] Can be done in parallel (read-only from cells)
- [ ] Upload to GPU

### Debug Rendering

- [ ] Icosphere meshes for cells (same as Preview)
- [ ] Instanced rendering for all active cells
- [ ] Line rendering for adhesions
- [ ] Gizmos for selected cells
- [ ] World boundary sphere

### LOD System (Optional for CPU mode)

- [ ] With 10K cells, may benefit from LOD
- [ ] Distance-based LOD levels
- [ ] Lower poly meshes for distant cells
- [ ] Cull cells beyond certain distance

### Cell Selection

- [ ] Mouse raycast (single-threaded, on render thread)
- [ ] Ray-sphere intersection test
- [ ] Use spatial grid for acceleration
- [ ] Highlight selected cell

### Performance Considerations

- [ ] Rendering separate from physics
- [ ] Physics tick rate independent of frame rate
- [ ] Target 60 FPS rendering, 50 TPS physics
- [ ] Don't block physics waiting for render

---

## UI Integration

### Time Scrubber

- [ ] Slider for manual tick control
- [ ] Pause simulation on scrub
- [ ] Jump to specific tick (requires state saving)
- [ ] Display current tick and time
- [ ] Forward/backward stepping

### State Saving for Scrubbing

- [ ] Save complete simulation state at intervals
- [ ] Position, velocity, rotation, angular velocity
- [ ] Mass, age, split count
- [ ] Adhesions and anchors
- [ ] Genome references
- [ ] Compress states to save memory
- [ ] Limit state history (e.g., last 1000 ticks)

### Pause/Play/Step Controls

- [ ] Pause button (set paused flag atomically)
- [ ] Play button (clear paused flag)
- [ ] Step forward one tick
- [ ] Step backward one tick (from state history)
- [ ] Reset to tick 0

### Simulation Info Display

- [ ] Current tick
- [ ] Current time (seconds)
- [ ] Cell count (current / max)
- [ ] Adhesion count
- [ ] FPS (frames per second)
- [ ] TPS (ticks per second)
- [ ] Performance: Frieza units
- [ ] Thread utilization (optional)
- [ ] Physics breakdown (collision, adhesion, integration times)

### Selected Cell Info Panel

- [ ] Same as Preview
- [ ] Cell properties display
- [ ] Mode and genome info
- [ ] Adhesion list

### Multi-Cell Selection (Optional)

- [ ] Select multiple cells (shift-click, drag box)
- [ ] Display aggregate info (count, total mass)
- [ ] Operate on selection (delete, modify)

---

## Performance Optimization

### Multi-Threading Optimization

#### Thread Count Tuning

- [ ] Measure performance with different thread counts
- [ ] Find optimal count (often = physical cores)
- [ ] Hyperthreading may or may not help
- [ ] Configurable by user

#### Work Partitioning

- [ ] Balance load across threads
- [ ] Minimize thread idle time
- [ ] Static partitioning for regular work
- [ ] Dynamic for irregular work (divisions)

#### Cache Optimization

- [ ] SoA layout for cache efficiency
- [ ] Minimize false sharing (padding between thread data)
- [ ] Keep hot data together
- [ ] Thread-local buffers to avoid cache bouncing

#### Lock Contention

- [ ] Minimize locked regions
- [ ] Use lock-free algorithms where possible
- [ ] Per-thread data structures
- [ ] Read-write locks for read-heavy data

### Spatial Grid Optimization

- [ ] Tune grid cell size (balance occupancy vs grid size)
- [ ] With 10K cells in 200³ volume, many cells per grid cell
- [ ] Parallel grid rebuild if bottleneck
- [ ] Hierarchical grid (optional)

### Collision Detection Optimization

- [ ] Early exit for distant cells
- [ ] Broad phase via spatial grid
- [ ] Narrow phase only for nearby pairs
- [ ] SIMD for distance calculations (optional)
- [ ] Avoid redundant pair checks

### Adhesion Optimization

- [ ] Adhesions are main workload (5-10 per cell = 50K+ adhesions)
- [ ] Highly parallel (independent calculations)
- [ ] Cache-friendly access pattern
- [ ] Avoid unnecessary calculations (skip inactive adhesions)

### Memory Optimization

- [ ] Pre-allocate arrays to max capacity
- [ ] Avoid dynamic allocations during physics step
- [ ] Reuse thread-local buffers
- [ ] Compact arrays periodically (not every tick)

### Profiling Points

- [ ] Grid rebuild time
- [ ] Collision detection time
- [ ] Adhesion force calculation time
- [ ] Integration time
- [ ] Division time
- [ ] Mass transfer time
- [ ] Total physics step time
- [ ] Thread synchronization overhead
- [ ] Thread utilization (% time working vs idle)

### Performance Target Validation

- [ ] Measure with 5K cells
- [ ] Measure with 10K cells
- [ ] Ensure < 2Fr (< 0.4ms per tick at 10K cells)
- [ ] Profile bottlenecks
- [ ] Optimize critical paths
- [ ] Achieve target 50+ TPS

---

## Testing & Validation

### Unit Tests

#### Multi-Threading Tests

- [ ] Thread pool initialization
- [ ] Work partitioning correctness
- [ ] Force accumulation (all strategies)
- [ ] Barrier synchronization
- [ ] No data races (use sanitizers)
- [ ] Lock-free data structure correctness

#### Physics Tests (Same as Preview)

- [ ] Verlet integration
- [ ] Collision detection
- [ ] Collision response
- [ ] Boundary forces
- [ ] Adhesion forces
- [ ] Cell division
- [ ] Mass transfer

### Integration Tests

#### Simulation Tests

- [ ] Run 1000 ticks without crashes
- [ ] Energy conservation (approximate, not perfect)
- [ ] Momentum conservation (approximate)
- [ ] Cell division produces correct cell count
- [ ] Adhesions maintained after division
- [ ] No deadlocks in multi-threaded execution

#### Concurrency Tests

- [ ] Run same simulation with 1, 2, 4, 8 threads
- [ ] Results should be similar (not identical due to floating-point order)
- [ ] No crashes or data races
- [ ] Performance scales with thread count (up to a point)

#### State Saving Tests

- [ ] Save state at tick N
- [ ] Load state and continue
- [ ] Compare with direct run to tick N+M
- [ ] Results should be similar

### Parity Tests

#### Preview vs CPU Parity

- [ ] Run same genome in Preview and CPU modes
- [ ] Cell count matches at each tick (approximately)
- [ ] Organism shape similar (not identical)
- [ ] Energy/mass balance similar
- [ ] Acceptable divergence over time

### Performance Tests

- [ ] Measure tick time with 1K, 5K, 10K cells
- [ ] Verify < 2Fr at 10K cells
- [ ] Measure scalability with thread count
- [ ] Identify bottlenecks
- [ ] Optimize critical sections

### Stress Tests

- [ ] 10K cells all dividing simultaneously
- [ ] 10K cells with maximum adhesions
- [ ] 10K cells clustered (collision stress)
- [ ] 10K cells spread across world
- [ ] Rapid genome changes
- [ ] Long-duration runs (hours)

### Stability Tests

- [ ] Run for 100K+ ticks
- [ ] No energy explosion
- [ ] No momentum drift
- [ ] No crashes
- [ ] No memory leaks
- [ ] No performance degradation over time

### Thread Safety Validation

- [ ] Run with ThreadSanitizer (TSan)
- [ ] Run with AddressSanitizer (ASan)
- [ ] Valgrind helgrind (data race detection)
- [ ] No warnings or errors

---

## Implementation Phases

### Phase 1: Foundation

- [ ] Set up CPU simulation module structure
- [ ] Implement core data structures (cells, adhesions, genome)
- [ ] Implement spatial grid (sequential version)
- [ ] Basic cell allocation (add, remove)
- [ ] Simple rendering (debug icospheres)
- [ ] Initialize thread pool

### Phase 2: Sequential Physics

- [ ] Velocity Verlet integration (single-threaded first)
- [ ] Collision detection (using spatial grid)
- [ ] Collision response forces
- [ ] World boundary forces
- [ ] Test with a few hundred cells
- [ ] Verify correctness before parallelizing

### Phase 3: Parallel Force Calculation

- [ ] Implement force accumulation strategy (per-thread buffers recommended)
- [ ] Parallelize collision detection
- [ ] Parallelize boundary force calculation
- [ ] Barrier synchronization
- [ ] Test with multiple threads
- [ ] Verify no data races (use TSan)

### Phase 4: Adhesions

- [ ] Adhesion data structure
- [ ] Adhesion force calculations (sequential first)
- [ ] Parallelize adhesion calculations
- [ ] Force accumulation from adhesions
- [ ] Adhesion rendering (lines)
- [ ] Test with manually created adhesions

### Phase 5: Cell Division

- [ ] Division criteria checking (parallel)
- [ ] Division execution (single-threaded recommended)
- [ ] Mass splitting
- [ ] Position/orientation calculation
- [ ] Adhesion inheritance (zones A, B, C)
- [ ] Geometric anchor recalculation
- [ ] Test with simple organisms

### Phase 6: Genome Integration

- [ ] Thread-safe genome library (Arc<RwLock<>>)
- [ ] Connect to genome editor
- [ ] Spawn initial cells from genome
- [ ] Hot-reload on genome change
- [ ] Validate genome structure
- [ ] Test with various genomes

### Phase 7: Energy System

- [ ] Mass accumulation (parallel)
- [ ] Mass consumption (parallel)
- [ ] Mass transfer between cells (handle write conflicts)
- [ ] Cell death and removal
- [ ] Test energy conservation (approximate)

### Phase 8: UI Integration

- [ ] Time scrubber control
- [ ] State saving for scrubbing
- [ ] Pause/play/step controls
- [ ] Info displays (tick, cell count, performance)
- [ ] Cell selection (raycasting)
- [ ] Selected cell info panel
- [ ] Test all UI interactions

### Phase 9: Performance Optimization

- [ ] Profile with 5K and 10K cells
- [ ] Optimize bottlenecks
- [ ] Tune thread count
- [ ] Optimize grid rebuild
- [ ] Optimize force accumulation
- [ ] Achieve < 2Fr target
- [ ] Verify scalability

### Phase 10: Polish and Testing

- [ ] Comprehensive testing (concurrency, stability, stress)
- [ ] Bug fixes
- [ ] Improve rendering quality
- [ ] Add gizmos and visual aids
- [ ] Thread sanitizer validation
- [ ] Parity testing with Preview mode

### Phase 11: Advanced Features (Post-MVP)

- [ ] Cell types (passive, active, sensors)
- [ ] Signal system (if implemented)
- [ ] Grid-based features (chemicals, food, light)
- [ ] Advanced rendering (LOD, culling)

---

## Known Differences from Preview

### Determinism

- **Preview:** Bit-for-bit deterministic (single-threaded)
- **CPU:** Approximate determinism (floating-point operation order varies)
- **Impact:** Same genome produces similar but not identical organisms
- **Acceptable:** As long as long-term stability maintained

### Performance Characteristics

- **Preview:** Optimized for low cell count (256), fast iteration
- **CPU:** Optimized for medium cell count (10K), parallelism
- **Trade-off:** More complexity for better throughput

### Thread Safety

- **Preview:** No concurrency issues (single-threaded)
- **CPU:** Careful synchronization required
- **Complexity:** Higher implementation and testing burden

---

## Success Criteria

### Functional Requirements Met

- [ ] 10K cells simulated efficiently
- [ ] Multi-threaded execution without data races
- [ ] Velocity Verlet integration at 50 TPS
- [ ] Cell division with adhesion inheritance
- [ ] Genome hot-reload and validation
- [ ] Time scrubber with state saving
- [ ] Cell selection and info display
- [ ] Similar results to Preview mode (approximate parity)

### Performance Requirements Met

- [ ] < 2Fr (10K cells at >50 TPS)
- [ ] Smooth rendering at 60 FPS
- [ ] Efficient thread utilization (>80% when running)
- [ ] No performance degradation over time
- [ ] Scalable with thread count (2-8 threads)

### Quality Requirements Met

- [ ] Approximate determinism (similar results across runs)
- [ ] Energy conservation (approximate, no explosions)
- [ ] Momentum conservation (approximate)
- [ ] Stable long-term simulation (hours without issues)
- [ ] No data races (verified with TSan)
- [ ] No memory leaks (verified with ASan/Valgrind)
- [ ] Accurate organism development (matches genome closely)

---

## Notes

### Design Decisions

#### Multi-Threading Architecture

- **Thread pool:** Use rayon or custom pool
- **Work partitioning:** Hybrid static/dynamic approach
- **Synchronization:** Minimize barriers, use lock-free where possible
- **Force accumulation:** Per-thread buffers recommended

#### Determinism Trade-Off

- Perfect determinism not required (unlike Preview)
- Floating-point operation order varies with threading
- Results should be "close enough" for practical purposes
- Long-term stability more important than bit-for-bit accuracy

#### State Saving

- Essential for scrubber functionality
- Save states at regular intervals (e.g., every 10 ticks)
- Compress states to reduce memory usage
- Limit history to prevent unbounded growth

### Common Pitfalls to Avoid

- **Data races:** Use TSan early and often
- **Deadlocks:** Minimize locks, avoid circular dependencies
- **False sharing:** Pad thread-local data to cache line boundaries
- **Excessive synchronization:** Balance safety vs performance
- **Irregular load distribution:** Monitor thread utilization
- **Memory leaks:** Validate with ASan/Valgrind
- **Lock contention:** Profile and optimize hot locks

### Performance Considerations

- **Adhesion calculation:** Usually the bottleneck (50K+ adhesions)
- **Grid rebuild:** Can be expensive, consider optimizations
- **Thread synchronization:** Barriers add overhead
- **Cache misses:** SoA layout helps, but watch access patterns
- **Scalability limits:** Amdahl's law applies, some work is sequential

### Reference Implementations

- Preview simulation (reference for physics correctness)
- C++ version (reference for adhesion mechanics)
- Rayon parallel iterators (best practices for data parallelism)

---

## Changelog

**v1.0** - Initial comprehensive checklist

- All core systems documented
- Multi-threading architecture specified
- Implementation phases planned
- Success criteria defined
- Differences from Preview mode noted