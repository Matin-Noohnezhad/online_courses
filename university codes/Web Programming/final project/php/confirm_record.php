<?php

// get the q parameter from URL
$q = $_REQUEST["q"];
$w = $_REQUEST["w"];

$servername = "localhost";
$user = "root";
$password = "";
$dbname = "final_project";

// Create connection
$conn = new mysqli($servername, $user, $password, $dbname);
mysqli_set_charset($conn, "utf8");
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$exist = false;
$sql = "UPDATE good SET confirmed=" . $w . " where id = " . $q;
$result = $conn->query($sql);

?>
