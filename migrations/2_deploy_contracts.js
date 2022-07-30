var store = artifacts.require("./contracts/decentraldl.sol");

module.exports = function(deployer) {
  deployer.deploy(store);
};