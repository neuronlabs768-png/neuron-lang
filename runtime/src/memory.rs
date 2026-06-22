/// NEURON Memory allocator — arena-based, tensor-aware.
///
/// Allocates tensor data in arenas for cache-friendly access.
/// Gradient buffers are allocated adjacent to their forward tensors.

use std::alloc::{alloc, dealloc, Layout};

/// Arena allocator for tensor data.
pub struct Arena {
    blocks: Vec<ArenaBlock>,
    current_block: usize,
    offset: usize,
    block_size: usize,
}

struct ArenaBlock {
    ptr: *mut u8,
    layout: Layout,
    capacity: usize,
}

impl Arena {
    /// Create a new arena with the given block size (default 4MB).
    pub fn new(block_size: usize) -> Self {
        let mut arena = Self {
            blocks: Vec::new(),
            current_block: 0,
            offset: 0,
            block_size,
        };
        arena.alloc_block();
        arena
    }

    pub fn default() -> Self {
        Self::new(4 * 1024 * 1024) // 4MB blocks
    }

    fn alloc_block(&mut self) {
        let layout = Layout::from_size_align(self.block_size, 64).unwrap();
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            panic!("Arena: out of memory (failed to allocate {} bytes)", self.block_size);
        }
        self.blocks.push(ArenaBlock { ptr, layout, capacity: self.block_size });
        self.current_block = self.blocks.len() - 1;
        self.offset = 0;
    }

    /// Allocate `n` f64 values from the arena. Returns a mutable slice.
    pub fn alloc_f64(&mut self, n: usize) -> &mut [f64] {
        let byte_size = n * std::mem::size_of::<f64>();
        let align = std::mem::align_of::<f64>();

        // Align offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + byte_size > self.block_size {
            // Need a new block
            if byte_size > self.block_size {
                // Oversized allocation — create a custom block
                let layout = Layout::from_size_align(byte_size, 64).unwrap();
                let ptr = unsafe { alloc(layout) };
                if ptr.is_null() { panic!("Arena: out of memory"); }
                self.blocks.push(ArenaBlock { ptr, layout, capacity: byte_size });
                return unsafe { std::slice::from_raw_parts_mut(ptr as *mut f64, n) };
            }
            self.alloc_block();
            return self.alloc_f64(n);
        }

        let block = &self.blocks[self.current_block];
        let ptr = unsafe { block.ptr.add(aligned_offset) as *mut f64 };
        self.offset = aligned_offset + byte_size;
        unsafe { std::slice::from_raw_parts_mut(ptr, n) }
    }

    /// Total bytes allocated across all blocks.
    pub fn total_allocated(&self) -> usize {
        self.blocks.iter().map(|b| b.capacity).sum()
    }

    /// Reset the arena — all previous allocations are invalidated.
    pub fn reset(&mut self) {
        self.current_block = 0;
        self.offset = 0;
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for block in &self.blocks {
            unsafe { dealloc(block.ptr, block.layout); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc() {
        let mut arena = Arena::new(1024);
        let slice = arena.alloc_f64(10);
        assert_eq!(slice.len(), 10);
        for i in 0..10 { slice[i] = i as f64; }
        assert_eq!(slice[5], 5.0);
    }

    #[test]
    fn test_arena_multiple() {
        let mut arena = Arena::new(256);
        let a = arena.alloc_f64(8);
        let a_ptr = a.as_mut_ptr();
        let a_len = a.len();
        let b = arena.alloc_f64(8);
        assert_eq!(a_len, 8);
        assert_eq!(b.len(), 8);
        // Verify they don't overlap by writing through raw pointer
        unsafe {
            *a_ptr = 1.0;
            b[0] = 2.0;
            assert_eq!(*a_ptr, 1.0);
            assert_eq!(b[0], 2.0);
        }
    }

    #[test]
    fn test_arena_overflow() {
        let mut arena = Arena::new(128);
        // Should trigger new block allocation
        let a = arena.alloc_f64(100);
        assert_eq!(a.len(), 100);
    }
}
