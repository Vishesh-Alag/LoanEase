var currentTab = 0;
showTab(currentTab);

function showTab(n) {
  var x = document.getElementsByClassName("tab");
  x[n].style.display = "block";
  if (n == 0) {
    document.getElementById("prevBtn").style.display = "none";
  } else {
    document.getElementById("prevBtn").style.display = "inline";
  }
  if (n == (x.length - 1)) {
    document.getElementById("nextBtn").innerHTML = "Submit";
  } else {
    document.getElementById("nextBtn").innerHTML = "Next";
  }
  fixStepIndicator(n)
}

function nextPrev(n) {
  var x = document.getElementsByClassName("tab");
  if (n == 1 && !validateForm()) return false;
  x[currentTab].style.display = "none";
  currentTab = currentTab + n;
  if (currentTab >= x.length) {
    document.getElementById("multiStepForm").submit();
    return false;
  }
  showTab(currentTab);
}

function validateForm() {
  var x, y, i, valid = true;
  x = document.getElementsByClassName("tab");
  y = x[currentTab].getElementsByTagName("input");
  for (i = 0; i < y.length; i++) {
    if (y[i].value == "" && y[i].hasAttribute("required")) {
      y[i].className += " invalid";
      valid = false;
    }
  }
  if (valid) {
    document.getElementsByClassName("step")[currentTab].className += " finish";
  }
  return valid;
}

function fixStepIndicator(n) {
  var i, x = document.getElementsByClassName("step");
  for (i = 0; i < x.length; i++) {
    x[i].className = x[i].className.replace(" active", "");
  }
  x[n].className += " active";
}

function updateResidentialAssetValue() {
  var residentialAsset = document.getElementById("residentialAsset").value;
  var residentialAssetValueInput = document.getElementById("residentialAssetValue");
  if (residentialAsset === "Rented") {
    residentialAssetValueInput.value = "0";
    residentialAssetValueInput.disabled = true;
  } else {
    residentialAssetValueInput.value = "";
    residentialAssetValueInput.disabled = false;
  }
}

function updateCommercialAssetValue() {
  var commercialAsset = document.getElementById("commercialAsset").value;
  var commercialAssetValueInput = document.getElementById("commercialAssetValue");
  if (commercialAsset === "Rented") {
    commercialAssetValueInput.value = "0";
    commercialAssetValueInput.disabled = true;
  } else {
    commercialAssetValueInput.value = "";
    commercialAssetValueInput.disabled = false;
  }
}

function toggleLuxuryAssetField() {
  var dropdownValue = document.getElementById("luxuryAssetDropdown").value;
  var container = document.getElementById("luxuryAssetsContainer");
  if (dropdownValue === "Yes") {
    container.style.display = "block";
  } else {
    container.style.display = "none";
  }
}

function addLuxuryAssetField() {
  var container = document.getElementById("luxuryAssetsContainer");
  var newDiv = document.createElement("div");
  newDiv.innerHTML = '<p><input type="text" placeholder="Luxury Asset Type" oninput="this.className = \'\'"></p><p><input type="number" placeholder="Valuation (in integers)" oninput="this.className = \'\'"></p><button type="button" onclick="removeLuxuryAssetField(this)">- Remove</button>';
  container.appendChild(newDiv);
}

function removeLuxuryAssetField(element) {
  element.parentNode.remove();
}
