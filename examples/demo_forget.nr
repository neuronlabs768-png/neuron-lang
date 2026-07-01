// ═══════════════════════════════════════════════════════════════
//  DEMO 3: "Machine Unlearning — Built Into the Language"
//  NEURON can selectively forget learned knowledge and issue
//  a cryptographic ForgetCertificate proving it was done.
// ═══════════════════════════════════════════════════════════════
//
//  Use case: A hospital trained a model on patient data.
//  Patient requests deletion under GDPR Article 17.
//  The hospital must PROVE the model no longer contains
//  that patient's information.
//
//  In Python: no standard mechanism. Manual retraining.
//  In NEURON: forget() with a verifiable certificate.
// ═══════════════════════════════════════════════════════════════

model DiagnosisModel:
  w: Tensor[4, 1] = glorot(4, 1)

  fn predict(self, symptoms: Tensor[B, 4]) -> Tensor[B, 1]:
    return sigmoid(symptoms @ self.w)

fn main() [Effect[Mut[net]]]:
  let net = DiagnosisModel()
  let patient_data = zeros(10, 4) + 1.0
  // Train the model (it learns from patient data)
  let pred1 = net.predict(patient_data)
  let loss1 = mse(pred1, zeros(10, 1) + 1.0)
  print(loss1)
  update net.w by sgd(grad(loss1), lr=0.5)

  let pred2 = net.predict(patient_data)
  let loss2 = mse(pred2, zeros(10, 1) + 1.0)
  print(loss2)
  update net.w by sgd(grad(loss2), lr=0.5)

  let pred3 = net.predict(patient_data)
  let loss3 = mse(pred3, zeros(10, 1) + 1.0)
  print(loss3)
  update net.w by sgd(grad(loss3), lr=0.5)

  // Patient requests data deletion under GDPR.
  // forget() removes the learned information and issues a certificate.
  let certificate = forget(net, patient_data, "FisherScrubbing", 0.5)

  // The certificate proves:
  //   - What method was used (GradientAscent / TaskNegation)
  //   - The strength of forgetting applied
  //   - Whether residual capability bounds are satisfied
  //   - A unique certificate ID for audit trail
  print(certificate)

  return certificate
