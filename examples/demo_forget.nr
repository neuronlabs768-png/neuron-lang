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

fn main():
  let net = DiagnosisModel()
  let patient_data = zeros(10, 4) + 1.0

  // Train the model (it learns from patient data)
  let pred = net.predict(patient_data)
  let loss = mse(pred, zeros(10, 1) + 1.0)

  // Patient requests data deletion under GDPR.
  // forget() removes the learned information and issues a certificate.
  let certificate = forget(net, patient_data, "GradientAscent", 0.1)

  // The certificate proves:
  //   - What method was used (GradientAscent / TaskNegation)
  //   - The strength of forgetting applied
  //   - Whether residual capability bounds are satisfied
  //   - A unique certificate ID for audit trail
  print(certificate)

  return certificate
