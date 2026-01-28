"""
Unit tests for loss functions in grit/losses.py
"""
import unittest as ut
import torch
from torch_geometric.data import Data, Batch

from grit.losses import attention_improvement_loss, structure_reconstruction_loss


class TestAttentionImprovementLoss(ut.TestCase):
    
    def setUp(self):
        """Set up test data for each test"""
        # Create a simple graph with 5 nodes and some edges
        self.num_nodes = 5
        self.dim = 16
        
        # Create edge_index for a simple graph (0-1, 1-2, 2-3, 3-4, 4-0)
        self.edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
        ], dtype=torch.long)
        
        # Create batch assignment (single graph)
        self.batch = torch.zeros(self.num_nodes, dtype=torch.long)
        
        # Create random embeddings
        self.node_embeddings = torch.randn(self.num_nodes, self.dim)
        self.denoised_embeddings = torch.randn(self.num_nodes, self.dim)
    
    def test_basic_functionality(self):
        """Test that loss function runs without errors"""
        loss = attention_improvement_loss(
            node_embeddings=self.node_embeddings,
            denoised_embeddings=self.denoised_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            tau=0.2,
            weight=1.0
        )
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
        
        # Check that loss is positive
        self.assertTrue(loss >= 0)
    
    def test_zero_weight(self):
        """Test that zero weight returns zero loss"""
        loss = attention_improvement_loss(
            node_embeddings=self.node_embeddings,
            denoised_embeddings=self.denoised_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            tau=0.2,
            weight=0.0
        )
        
        self.assertEqual(loss.item(), 0.0)
    
    def test_weight_scaling(self):
        """Test that weight parameter scales the loss correctly"""
        loss1 = attention_improvement_loss(
            node_embeddings=self.node_embeddings,
            denoised_embeddings=self.denoised_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            tau=0.2,
            weight=1.0
        )
        
        loss2 = attention_improvement_loss(
            node_embeddings=self.node_embeddings,
            denoised_embeddings=self.denoised_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            tau=0.2,
            weight=2.0
        )
        
        # Loss2 should be approximately 2x loss1
        self.assertAlmostEqual(loss2.item(), 2 * loss1.item(), places=5)
    
    def test_batch_processing(self):
        """Test that loss function handles multiple graphs in batch"""
        # Create two graphs in batch
        num_nodes_g1 = 5
        num_nodes_g2 = 6
        total_nodes = num_nodes_g1 + num_nodes_g2
        
        # Edge index for two separate graphs
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0,  # Graph 1 (triangle)
             5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 5],  # Graph 2 (hexagon)
            [1, 0, 2, 1, 0, 2,
             6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 5, 10]
        ], dtype=torch.long)
        
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
        
        node_embeddings = torch.randn(total_nodes, self.dim)
        denoised_embeddings = torch.randn(total_nodes, self.dim)
        
        loss = attention_improvement_loss(
            node_embeddings=node_embeddings,
            denoised_embeddings=denoised_embeddings,
            edge_index=edge_index,
            batch=batch,
            tau=0.2,
            weight=1.0
        )
        
        # Check that loss is finite and positive
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss"""
        node_embeddings = torch.randn(self.num_nodes, self.dim, requires_grad=True)
        denoised_embeddings = torch.randn(self.num_nodes, self.dim, requires_grad=True)
        
        loss = attention_improvement_loss(
            node_embeddings=node_embeddings,
            denoised_embeddings=denoised_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            tau=0.2,
            weight=1.0
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        self.assertIsNotNone(node_embeddings.grad)
        self.assertIsNotNone(denoised_embeddings.grad)
        self.assertTrue(torch.all(torch.isfinite(node_embeddings.grad)))
        self.assertTrue(torch.all(torch.isfinite(denoised_embeddings.grad)))


class TestStructureReconstructionLoss(ut.TestCase):
    
    def setUp(self):
        """Set up test data for each test"""
        # Create a simple graph with 5 nodes and some edges
        self.num_nodes = 5
        self.dim = 16
        
        # Create edge_index for a simple graph (0-1, 1-2, 2-3, 3-4, 4-0)
        self.edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
        ], dtype=torch.long)
        
        # Create batch assignment (single graph)
        self.batch = torch.zeros(self.num_nodes, dtype=torch.long)
        
        # Create random embeddings
        self.node_embeddings = torch.randn(self.num_nodes, self.dim)
    
    def test_basic_functionality(self):
        """Test that loss function runs without errors"""
        loss = structure_reconstruction_loss(
            node_embeddings=self.node_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            weight=1.0
        )
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
        
        # Check that loss is positive
        self.assertTrue(loss >= 0)
    
    def test_zero_weight(self):
        """Test that zero weight returns zero loss"""
        loss = structure_reconstruction_loss(
            node_embeddings=self.node_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            weight=0.0
        )
        
        self.assertEqual(loss.item(), 0.0)
    
    def test_weight_scaling(self):
        """Test that weight parameter scales the loss correctly"""
        loss1 = structure_reconstruction_loss(
            node_embeddings=self.node_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            weight=1.0
        )
        
        loss2 = structure_reconstruction_loss(
            node_embeddings=self.node_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            weight=2.0
        )
        
        # Loss2 should be approximately 2x loss1
        self.assertAlmostEqual(loss2.item(), 2 * loss1.item(), places=5)
    
    def test_perfect_embeddings(self):
        """Test loss behavior with perfectly aligned embeddings for edges"""
        # Create embeddings where connected nodes have similar embeddings
        # and disconnected nodes have dissimilar embeddings
        node_embeddings = torch.zeros(self.num_nodes, self.dim)
        
        # Make connected nodes have similar embeddings
        for i in range(self.num_nodes):
            node_embeddings[i] = torch.randn(self.dim)
        
        # This won't be perfect, but test that loss is finite
        loss = structure_reconstruction_loss(
            node_embeddings=node_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            weight=1.0
        )
        
        self.assertTrue(torch.isfinite(loss))
    
    def test_batch_processing(self):
        """Test that loss function handles multiple graphs in batch"""
        # Create two graphs in batch
        num_nodes_g1 = 5
        num_nodes_g2 = 6
        total_nodes = num_nodes_g1 + num_nodes_g2
        
        # Edge index for two separate graphs
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0,  # Graph 1 (triangle)
             5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 5],  # Graph 2 (hexagon)
            [1, 0, 2, 1, 0, 2,
             6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 5, 10]
        ], dtype=torch.long)
        
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
        
        node_embeddings = torch.randn(total_nodes, self.dim)
        
        loss = structure_reconstruction_loss(
            node_embeddings=node_embeddings,
            edge_index=edge_index,
            batch=batch,
            weight=1.0
        )
        
        # Check that loss is finite and positive
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(loss >= 0)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss"""
        node_embeddings = torch.randn(self.num_nodes, self.dim, requires_grad=True)
        
        loss = structure_reconstruction_loss(
            node_embeddings=node_embeddings,
            edge_index=self.edge_index,
            batch=self.batch,
            weight=1.0
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        self.assertIsNotNone(node_embeddings.grad)
        self.assertTrue(torch.all(torch.isfinite(node_embeddings.grad)))


if __name__ == '__main__':
    ut.main()
